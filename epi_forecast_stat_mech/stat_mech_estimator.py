# Lint as: python3
"""A high level modeling interface for jax-based statistical + mechanistic models."""
import collections
import dataclasses
import functools
import itertools
import logging
from typing import Callable

import jax
from jax.experimental import optimizers
import jax.numpy as jnp
import numpy as np
import xarray

import tensorflow_probability
tfp = tensorflow_probability.experimental.substrates.jax
tfd = tfp.distributions

from epi_forecast_stat_mech import data_model  # pylint: disable=g-bad-import-order
from epi_forecast_stat_mech import estimator_base  # pylint: disable=g-bad-import-order
from epi_forecast_stat_mech import mask_time  # pylint: disable=g-bad-import-order
from epi_forecast_stat_mech.mechanistic_models import mechanistic_models  # pylint: disable=g-bad-import-order
from epi_forecast_stat_mech.mechanistic_models import observables  # pylint: disable=g-bad-import-order
from epi_forecast_stat_mech.mechanistic_models import predict_lib  # pylint: disable=g-bad-import-order
from epi_forecast_stat_mech.statistical_models import base as stat_base  # pylint: disable=g-bad-import-order
from epi_forecast_stat_mech.statistical_models import network_models  # pylint: disable=g-bad-import-order


LogLikelihoods = collections.namedtuple(
    "LogLikelihoods",
    ["stat_log_prior",
     "mech_log_prior",
     "mech_log_likelihood",
     "stat_log_likelihood"])


@dataclasses.dataclass
class StatMechEstimator(estimator_base.Estimator):
  """A place-holder model that uses a mixed statistical/mechanistic approach."""

  train_steps: int
  stat_model: stat_base.StatisticalModel = dataclasses.field(
      default_factory=network_models.NormalDistributionModel)
  mech_model: mechanistic_models.MechanisticModel = dataclasses.field(
      default_factory=mechanistic_models.ViboudChowellModel)
  fused_train_steps: int = 100
  # TODO(mcoram): Resolve whether the "30" is deprecated hack or still useful.
  time_mask_fn: Callable[..., np.array] = functools.partial(
      mask_time.make_mask, min_value=30)
  fit_seed: int = 42
  observable_choice: observables.Observables = dataclasses.field(
      default_factory=lambda: observables.ObserveSpecified(["log_K"]))

  def _log_likelihoods(
      self,
      params,
      epidemics,
      covariates,
      mech_model,
      stat_model
  ):
    """Computes log-likelihood of the model parameters given epidemics data.

    The log likelihood computed here consists of two terms. First represents
    the likelihood of the observed epidemics trajectory under mechanistic model
    and the second represents the likelihood of the infered parameters given the
    covariates. In addition both terms include prior assumptions on the
    parameters of the models.

    Args:
      params: tuple of parameters for statistical and mechanistic models.
      epidemics: named tuple containing observed epidemics trajectory.
      covariates: array representing covariates for each location.
      mech_model: mechanistic model.
      stat_model: statistical model.

    Returns:
      namedtuple containing components of the negative log-likelihood.
    """
    stat_params, mech_params = params
    statistical_log_prior = stat_model.log_prior(stat_params)
    mechanistic_log_prior = jax.vmap(mech_model.log_prior)(mech_params)

    epidemic_observables_fn = jax.vmap(self.observable_choice.observables,
                                       [None, 0, 0])
    epidemic_observables = epidemic_observables_fn(mech_model, mech_params,
                                                   epidemics)
    self.epidemic_observables = epidemic_observables

    statistical_log_likelihood = stat_model.log_likelihood(
        stat_params, covariates, epidemic_observables)

    mech_log_likelihood_fn = jax.vmap(mech_model.log_likelihood)
    mechanistic_log_likelihood = mech_log_likelihood_fn(mech_params, epidemics)
    return LogLikelihoods(
        stat_log_prior=statistical_log_prior,
        mech_log_prior=mechanistic_log_prior,
        mech_log_likelihood=mechanistic_log_likelihood,
        stat_log_likelihood=statistical_log_likelihood)

  def fit(self, data):
    train_steps = self.train_steps
    seed = self.fit_seed
    data_model.validate_data_for_fit(data)
    # TODO(dkochkov) consider a tunable module for preprocessing.
    data["total"] = (
        ("location", "time",), np.cumsum(
            data.new_infections.transpose("location", "time").values, -1))
    num_locations = data.sizes["location"]
    self.data = data
    self.covariates = covariates = data.static_covariates.transpose(
        "location", "static_covariate").values
    self.epidemics = epidemics = (
        mechanistic_models.pack_epidemics_record_tuple(data))
    self.time_mask = self.time_mask_fn(data)

    # Mechanistic model initialization
    mech_params = jnp.stack([
        self.mech_model.init_parameters() for _ in range(num_locations)])
    epidemic_observables_fn = jax.vmap(self.observable_choice.observables,
                                       [None, 0, 0])
    epidemic_observables = epidemic_observables_fn(self.mech_model, mech_params,
                                                   epidemics)

    # Statistical model initialization
    rng = jax.random.PRNGKey(seed)
    stat_params = self.stat_model.init_parameters(rng, covariates,
                                                  epidemic_observables)
    init_params = (stat_params, mech_params)

    opt_init, opt_update, get_params = optimizers.adam(5e-4, b1=0.9, b2=0.999)
    opt_state = opt_init(init_params)

    @jax.value_and_grad
    def negative_log_prob(params):
      log_likelihoods = self._log_likelihoods(
          params, epidemics, covariates, self.mech_model, self.stat_model)
      mech_log_likelihood = log_likelihoods.mech_log_likelihood
      mech_log_likelihood = jnp.where(self.time_mask, mech_log_likelihood, 0.)
      log_likelihoods = log_likelihoods._replace(
          mech_log_likelihood=mech_log_likelihood)
      return -1. * sum(jax.tree_leaves(jax.tree_map(jnp.sum, log_likelihoods)))

    @jax.jit
    def train_step(step, opt_state):
      params = get_params(opt_state)
      loss_value, grad = negative_log_prob(params)
      opt_state = opt_update(step, grad, opt_state)
      return opt_state, loss_value

    # For some of these models (especially on accelerators), a single training
    # step runs very quickly. Fusing steps together considerably improves
    # performance.
    @functools.partial(jax.jit, static_argnums=(1,))
    def repeated_train_step(step, repeats, opt_state):
      def f(carray, _):
        step, opt_state, _ = carray
        opt_state, loss_value = train_step(step, opt_state)
        return (step + 1, opt_state, loss_value), None
      (_, opt_state, loss_value), _ = jax.lax.scan(
          f, (step, opt_state, 0.0), xs=None, length=repeats)
      return opt_state, loss_value

    for step in range(0, train_steps, self.fused_train_steps):
      opt_state, loss_value = repeated_train_step(
          step, self.fused_train_steps, opt_state)
      if step % 1000 == 0:
        logging.info(f"Loss at step {step} is: {loss_value}.")  # pylint: disable=logging-format-interpolation

    self.params_ = get_params(opt_state)
    self._is_trained = True
    return self

  def _check_fitted(self):
    if not hasattr(self, "params_"):
      raise AttributeError("`fit` must be called before `predict`.")

  def predict(self, test_data, num_samples, seed=0):
    self._check_fitted()
    rng = jax.random.PRNGKey(seed)

    _, mech_params = self.params_

    sample_mech_params_fn = getattr(
        self, "sample_mech_params_fn", lambda rngkey, num_samples: jnp.swapaxes(
            jnp.broadcast_to(mech_params,
                             (num_samples,) + mech_params.shape), 1, 0))

    return predict_lib.simulate_predictions(self.mech_model, mech_params,
                                            self.data, self.epidemics,
                                            test_data, num_samples, rng,
                                            sample_mech_params_fn)

  @property
  def mech_params(self):
    self._check_fitted()
    return predict_lib.mech_params_array(self.data, self.mech_model,
                                         self.params_[1])

  @property
  def encoded_mech_params(self):
    self._check_fitted()
    return predict_lib.encoded_mech_params_array(self.data, self.mech_model,
                                                 self.params_[1])

  # TODO(mcoram): Implement mech_params_hat.

  @property
  def alpha(self):
    self._check_fitted()
    if issubclass(self.stat_model.predict_module, network_models.LinearModule):
      kernel, unused_intercept = self.stat_model.linear_coefficients(
          self.params_[0])
      assert kernel.shape[1] == len(self.epidemic_observables.keys()), (
          f"unexpected kernel shape: {kernel.shape[1]} vs "
          f"{self.epidemic_observables.keys()}")
      alpha = xarray.DataArray(
          np.asarray(kernel),
          dims=("static_covariate", "encoded_param"),
          coords=dict(
              static_covariate=self.data.static_covariate,
              encoded_param=list(self.epidemic_observables.keys())))
      return alpha
    else:
      raise AttributeError("no alpha method for stat_model: %s" %
                           (self.stat_model.__class__,))

  @property
  def intercept(self):
    self._check_fitted()
    if issubclass(self.stat_model.predict_module, network_models.LinearModule):
      unused_kernel, bias = self.stat_model.linear_coefficients(
          self.params_[0])
      assert bias.shape == (len(self.epidemic_observables.keys()),), (
          f"unexpected bias shape: {bias.shape} vs "
          f"{self.epidemic_observables.keys()}")
      bias = xarray.DataArray(
          np.asarray(bias),
          dims=("encoded_param"),
          coords=dict(
              encoded_param=list(self.epidemic_observables.keys())))
      return bias
    else:
      raise AttributeError("no intercept method for stat_model: %s" %
                           (self.stat_model.__class__,))


def laplace_prior(parameters, scale_parameter=1.):
  return jax.tree_map(
      lambda x: tfd.Laplace(
          loc=jnp.zeros_like(x), scale=scale_parameter * jnp.ones_like(x)).
      log_prob(x), parameters)


def get_estimator_dict(
    train_steps=100000,
    fused_train_steps=100,
    time_mask_fn=functools.partial(mask_time.make_mask, min_value=50),
    fit_seed=42,
    list_of_prior_fns=(None, laplace_prior),
    list_of_mech_models=(mechanistic_models.ViboudChowellModel,
                         mechanistic_models.GaussianModel,
                         mechanistic_models.ViboudChowellModelPseudoLikelihood,
                         mechanistic_models.GaussianModelPseudoLikelihood,
                         mechanistic_models.StepBasedMultiplicativeGrowthModel,
                         mechanistic_models.StepBasedBaselineSEIRModel,
                         mechanistic_models.ViboudChowellModelPublished),
    list_of_stat_module=(network_models.LinearModule,
                         network_models.PerceptronModule),
    list_of_prior_names=("None", "Laplace"),
    list_of_mech_names=("VC", "Gaussian", "VC_PL", "Gaussian_PL",
                        "MultiplicativeGrowth", "BaselineSEIR", "VCPub"),
    list_of_stat_names=("Linear", "MLP"),
    list_of_observable_choices=(observables.InternalParams(),
                                observables.ObserveSpecified(["log_K"])),
    list_of_observable_choices_names=("ObsEnc", "ObsLogK")):

  # TODO(mcoram): Resolve whether the time_mask_value of "50" is deprecated
  # hack or still useful.
  # Create an iterator
  components_iterator = itertools.product(itertools.product(
      itertools.product(list_of_prior_fns, list_of_mech_models),
      list_of_stat_module), list_of_observable_choices)
  names_iterator = itertools.product(itertools.product(
      itertools.product(list_of_prior_names, list_of_mech_names),
      list_of_stat_names), list_of_observable_choices_names)

  # Combine into one giant dictionary of predictions
  estimator_dictionary = {}
  for components, name_components in zip(components_iterator, names_iterator):
    ((prior_fn, mech_model_cls), stat_module), observable_choice = components
    (((prior_name, mech_name), stat_name),
     observable_choice_name) = name_components
    name_list = [prior_name, mech_name, stat_name, observable_choice_name]
    # To preserve old names, I'm dropping ObsLogK from the name.
    if observable_choice_name == "ObsLogK":
      name_list.pop()
    model_name = "_".join(name_list)
    stat_model = network_models.NormalDistributionModel(
        predict_module=stat_module,
        log_prior_fn=prior_fn)
    mech_model = mech_model_cls()
    estimator_dictionary[model_name] = StatMechEstimator(
        train_steps=train_steps,
        stat_model=stat_model,
        mech_model=mech_model,
        fused_train_steps=fused_train_steps,
        time_mask_fn=time_mask_fn,
        fit_seed=fit_seed,
        observable_choice=observable_choice)
  estimator_dictionary["Laplace_VC_Linear_ObsChar1"] = StatMechEstimator(
      train_steps=train_steps,
      stat_model=network_models.NormalDistributionModel(
          predict_module=network_models.LinearModule,
          log_prior_fn=laplace_prior),
      mech_model=mechanistic_models.ViboudChowellModel(),
      fused_train_steps=fused_train_steps,
      time_mask_fn=time_mask_fn,
      fit_seed=fit_seed,
      observable_choice=observables.ObserveSpecified([
          "log_r", "log_a", "log_characteristic_time",
          "log_characteristic_height"
      ]))
  return estimator_dictionary
