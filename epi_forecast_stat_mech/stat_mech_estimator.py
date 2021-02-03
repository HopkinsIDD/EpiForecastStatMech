# Lint as: python3
"""A high level modeling interface for jax-based statistical + mechanistic models."""
import collections
import dataclasses
import functools
import itertools
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import xarray

import tensorflow_probability
tfp = tensorflow_probability.experimental.substrates.jax
tfd = tfp.distributions

from epi_forecast_stat_mech import data_model  # pylint: disable=g-bad-import-order
from epi_forecast_stat_mech import estimator_base  # pylint: disable=g-bad-import-order
from epi_forecast_stat_mech import mask_time  # pylint: disable=g-bad-import-order
from epi_forecast_stat_mech import optim_lib
from epi_forecast_stat_mech.mechanistic_models import mechanistic_models  # pylint: disable=g-bad-import-order
from epi_forecast_stat_mech.mechanistic_models import observables  # pylint: disable=g-bad-import-order
from epi_forecast_stat_mech.mechanistic_models import predict_lib  # pylint: disable=g-bad-import-order
from epi_forecast_stat_mech.statistical_models import base as stat_base  # pylint: disable=g-bad-import-order
from epi_forecast_stat_mech.statistical_models import network_models  # pylint: disable=g-bad-import-order
from epi_forecast_stat_mech.statistical_models import no_stat_model # pylint: disable=g-bad-import-order
from epi_forecast_stat_mech.statistical_models import probability


LogLikelihoods = collections.namedtuple(
    "LogLikelihoods",
    ["stat_log_prior",
     "mech_log_prior",
     "mech_log_likelihood",
     "stat_log_likelihood"])


LEARNING_RATE_DEFAULT = 5E-3


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
  preprocess_fn: Callable[..., np.array] = lambda x: x
  fit_seed: int = 42
  learning_rate: float = LEARNING_RATE_DEFAULT
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
    data = self.preprocess_fn(data)
    data_model.validate_data_for_fit(data)
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
    @jax.value_and_grad
    def negative_log_prob(params):
      log_likelihoods = self._log_likelihoods(
          params, epidemics, covariates, self.mech_model, self.stat_model)
      mech_log_likelihood = log_likelihoods.mech_log_likelihood
      mech_log_likelihood = jnp.where(self.time_mask, mech_log_likelihood, 0.)
      log_likelihoods = log_likelihoods._replace(
          mech_log_likelihood=mech_log_likelihood)
      return -1. * sum(jax.tree_leaves(jax.tree_map(jnp.sum, log_likelihoods)))

    adam_loop = optim_lib.get_adam_optim_loop(negative_log_prob, learning_rate=self.learning_rate)

    self.params_ = adam_loop(
        init_params,
        train_steps=self.train_steps,
        fused_train_steps=self.fused_train_steps)
    self._is_trained = True
    return self

  def _check_fitted(self):
    if not hasattr(self, "params_"):
      raise AttributeError("`fit` must be called before `predict`.")

  def predict(self, test_data, num_samples, seed=0):
    self._check_fitted()
    rng = jax.random.PRNGKey(seed)
    encoded_mech_params = self.mech_params_for_jax_code
    dynamic_covariates = predict_lib.prepare_dynamic_covariates(
        self.data, test_data)

    sample_mech_params_fn = getattr(
        self, "sample_mech_params_fn", lambda rngkey, num_samples: jnp.swapaxes(
            jnp.broadcast_to(encoded_mech_params,
                             (num_samples,) + encoded_mech_params.shape), 1, 0))

    return predict_lib.simulate_dynamic_predictions(
        self.mech_model, encoded_mech_params,
        self.data, self.epidemics, dynamic_covariates,
        num_samples, rng, sample_mech_params_fn)

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

  @property
  def mech_params_for_jax_code(self):
    return self.encoded_mech_params.values

  @property
  def epidemic_observables(self):
    self._check_fitted()
    _, mech_params = self.params_
    epidemic_observables_fn = jax.vmap(self.observable_choice.observables,
                                       [None, 0, 0])
    my_observables = epidemic_observables_fn(self.mech_model, mech_params,
                                             self.epidemics)
    return my_observables

  @property
  def observables_loc_scale_hat(self):
    self._check_fitted()
    stat_params, _ = self.params_
    return self.stat_model.get_loc_scale(stat_params, self.covariates,
                                         self.epidemic_observables)

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


def seven_day_time_smooth_helper_(x):
  if "time" in x.dims:
    return x.rolling(time=7, min_periods=4).mean()
  else:
    return x


def seven_day_time_smooth(data):
  return xarray.Dataset({
      key: seven_day_time_smooth_helper_(x)
      for key, x in data.data_vars.items()
  })


def const_covariates(data):
  del data["static_covariates"]
  del data["static_covariate"]
  data["static_covariates"] = xarray.DataArray(
      np.zeros((data.sizes["location"], 1)),
      dims=["location", "static_covariate"])
  return data


def get_estimator_dict(
    train_steps=100000,
    fused_train_steps=100,
    fit_seed=42,
    learning_rate=LEARNING_RATE_DEFAULT,
    list_of_prior_fns=(probability.log_soft_mixed_laplace_on_kernels,),
    list_of_mech_models=(
        mechanistic_models.ViboudChowellModel, mechanistic_models.GaussianModel,
        mechanistic_models.ViboudChowellModelPseudoLikelihood,
        mechanistic_models.GaussianModelPseudoLikelihood,
        mechanistic_models.StepBasedMultiplicativeGrowthModel,
        mechanistic_models.StepBasedSimpleMultiplicativeGrowthModel,
        mechanistic_models.StepBasedGeneralizedMultiplicativeGrowthModel,
        mechanistic_models.StepBasedBaselineSEIRModel,
        mechanistic_models.ViboudChowellModelPublished,
        mechanistic_models.TurnerModel),
    list_of_stat_module=(network_models.LinearModule,
                         network_models.PerceptronModule,
                         no_stat_model.Null),
    list_of_time_mask_fn=(functools.partial(mask_time.make_mask, min_value=50),
                          functools.partial(
                              mask_time.make_mask,
                              min_value=1,
                              recent_day_limit=6 * 7),
                          functools.partial(
                              mask_time.make_mask,
                              min_value=1,
                              recent_day_limit=4 * 7)),
    list_of_preprocess_fn=(lambda x: x, seven_day_time_smooth, const_covariates),
    list_of_prior_names=("LSML",),
    list_of_mech_names=("VC", "Gaussian", "VC_PL", "Gaussian_PL",
                        "MultiplicativeGrowth", "SimpleMultiplicativeGrowth",
                        "GeneralizedMultiplicativeGrowth", "BaselineSEIR", "VCPub",
                        "Turner"),
    list_of_stat_names=("Linear", "MLP", "Null"),
    list_of_observable_choices=(observables.InternalParams(),),
    list_of_observable_choices_names=("ObsEnc",),
    list_of_time_mask_fn_names=("50cases", "6wk", "4wk"),
    list_of_preprocess_fn_names=("Id", "7day", "ConstCov"),
    list_of_error_model_names=("full", "plugin")):

  # TODO(mcoram): Resolve whether the time_mask_value of "50" is deprecated
  # hack or still useful.
  # Create an iterator
  components_iterator = itertools.product(
      itertools.product(
          itertools.product(
              itertools.product(
                  itertools.product(
                      itertools.product(list_of_prior_fns, list_of_mech_models),
                      list_of_stat_module), list_of_observable_choices),
              list_of_time_mask_fn), list_of_preprocess_fn),
      list_of_error_model_names)
  names_iterator = itertools.product(
      itertools.product(
          itertools.product(
              itertools.product(
                  itertools.product(
                      itertools.product(list_of_prior_names,
                                        list_of_mech_names),
                      list_of_stat_names), list_of_observable_choices_names),
              list_of_time_mask_fn_names), list_of_preprocess_fn_names),
      list_of_error_model_names)

  # Combine into one giant dictionary of predictions
  estimator_dictionary = {}
  for components, name_components in zip(components_iterator, names_iterator):
    ((((((prior_fn, mech_model_cls), stat_module), observable_choice),
       time_mask_fn), preprocess_fn), error_model) = components
    ((((((prior_name, mech_name), stat_name), observable_choice_name),
       time_mask_fn_name), preprocess_fn_name),
     error_model_name) = name_components
    name_list = [prior_name, mech_name, stat_name]

    # Check for unhelpful combinations here
    # TODO(edklein): revisit if we should exclude more ConstCov models
    if stat_name == "Null" and error_model_name == "plugin": continue
    if stat_name == "Null" and preprocess_fn_name == "ConstCov": continue

    # To preserve old names, I'm dropping ObsLogK from the name.
    if observable_choice_name != "ObsLogK":
      name_list.append(observable_choice_name)
    # To preserve old names, I'm dropping 50cases from the name.
    if time_mask_fn_name != "50cases":
      name_list.append(time_mask_fn_name)
    # To preserve old names, I'm dropping Id from the name.
    if preprocess_fn_name != "Id":
      name_list.append(preprocess_fn_name)
    if error_model_name != "full":
      name_list.append(error_model_name)
    model_name = "_".join(name_list)
    mech_model = mech_model_cls()
    if error_model != error_model_name:
      raise ValueError(
          f"Expected agreement b/w error_model and error_model_name: "
          f"{error_model}, {error_model_name}"
      )
    if stat_name == "Null":
      # there must be a better way to do this
      stat_model = stat_module()
    elif error_model_name == "full":
      stat_model = network_models.NormalDistributionModel(
          predict_module=stat_module,
          log_prior_fn=prior_fn,
          scale_eps=1E-2,
          error_model="full")
    elif error_model_name == "plugin":
      stat_model = network_models.NormalDistributionModel(
          predict_module=stat_module,
          log_prior_fn=prior_fn,
          scale_eps=mech_model.bottom_scale,
          error_model="plugin")
    else:
      raise ValueError(f"Unexpected error_model_name: {error_model_name}")

    estimator_dictionary[model_name] = StatMechEstimator(
        train_steps=train_steps,
        stat_model=stat_model,
        mech_model=mech_model,
        fused_train_steps=fused_train_steps,
        time_mask_fn=time_mask_fn,
        preprocess_fn=preprocess_fn,
        fit_seed=fit_seed,
        observable_choice=observable_choice)
  estimator_dictionary["Laplace_VC_Linear_ObsChar1"] = StatMechEstimator(
      train_steps=train_steps,
      stat_model=network_models.NormalDistributionModel(
          predict_module=network_models.LinearModule,
          log_prior_fn=probability.laplace_prior),
      mech_model=mechanistic_models.ViboudChowellModel(),
      fused_train_steps=fused_train_steps,
      time_mask_fn=functools.partial(mask_time.make_mask, min_value=50),
      fit_seed=fit_seed,
      learning_rate=learning_rate,
      observable_choice=observables.ObserveSpecified([
          "log_r", "log_a", "log_characteristic_time",
          "log_characteristic_height"
      ]))
  return estimator_dictionary
