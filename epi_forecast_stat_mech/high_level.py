# Lint as: python3
"""A high level modeling interface for wrapping models & fitting procedures."""
import collections
import dataclasses
import functools
import logging
from typing import TypeVar

import jax
from jax.experimental import optimizers
import jax.numpy as jnp
import numpy as np
import xarray

from epi_forecast_stat_mech.evaluation import monte_carlo  # pylint: disable=g-bad-import-order
from epi_forecast_stat_mech.mechanistic_models import mechanistic_models  # pylint: disable=g-bad-import-order
from epi_forecast_stat_mech.statistical_models import base as stat_base  # pylint: disable=g-bad-import-order
from epi_forecast_stat_mech.statistical_models import network_models  # pylint: disable=g-bad-import-order

T = TypeVar("T")


class Estimator:
  """Abstract base class for the high level modeling interface.

  An implementation of Estimator wraps statistical and/or mechanistic
  epidemic models along with a fitting procedure to real-world or simulated
  data.

  The design is loosely based on the Scikit-Learn Esimator API, except argument
  and returns values are in the form of labeled ``xarray.Dataset`` objects. The
  idea is that a uniform interface should make it easier to run high level
  evaluations.

  Key methods for `Estimator` subclasses:
    __init__: the signature of ``__init__`` is allowed to vary between models.
      This is where any model specific hyperparameters should be set.
    fit: fit this estimator to data.
    predict: make predictions based on the fit estimator.

  TODO(shoyer): write down lists of canonical variable names that should/could
  appear in ``observations``, ``covariates`` and ``predictions``.
  """

  def fit(self: T, observations: xarray.Dataset) -> T:  # pylint: disable=unused-argument
    """Fit this Estimator based on observations.

    A typical ``fit`` method might do something like the following:
    1. Convert ``observations`` from xarray.Dataset objects into "design
       matrices" of NumPy array(s).
    2. Run some sort of optimization/fitting procedure to minimize the loss of
       statistical and/or mechanicistic models.
    3. Save optimal parameters on the estimator object.

    Args:
      observations: a dataset of observed variables, including (at least?)
        dimensions "location" and "time". "time" should be real-world time
        (i.e., datetime64 dtype), not time since the start of the epidemic.

    Returns:
      This estimator object.
    """
    return self

  def predict(self, seed, time_steps, num_samples):
    """Make predictions from this estimator.

    The implementation of typical ``predict`` method would sample from the
    same underlying probabilistic models used in the ``fit`` step.

    The behavior of `predict` is not well defined if `fit` has not been called
    first.

    Args:
      seed: an integer seed for generating rollouts.
      time_steps: the length of the trajectories to generate.
      num_samples: the number of trajectories to generate for each location.

    Returns:
      An xarray of predictions. The axes are `location`, `sample` and `time`.
    """
    raise NotImplementedError


_EpidemicsRecord = collections.namedtuple(
    "_EpidemicsRecord",
    ["t", "infections_over_time", "cumulative_infections"])


LogLikelihoods = collections.namedtuple(
    "LogLikelihoods",
    ["stat_log_prior",
     "mech_log_prior",
     "mech_log_likelihood",
     "stat_log_likelihood"])


def _pack_epidemics_record_tuple(ds):
  return _EpidemicsRecord(
      np.tile(np.expand_dims(ds.time.values.astype(np.float32), 0),
              (ds.dims["location"], 1)),
      ds.new_infections.transpose("location", "time").values.astype(np.float32),
      np.cumsum(ds.new_infections.transpose("location", "time").values.astype(
          np.float32), axis=-1))


def _get_time_mask(ds, min_value=30):
  """Masks times while total number of infections is less than `min_value`."""
  total = ds.total.values
  mask = np.asarray(total > min_value).astype(np.float32)
  return mask


@dataclasses.dataclass
class StatMechEstimator(Estimator):
  """A place-holder model that uses a mixed statistical/mechanistic approach."""

  stat_model: stat_base.StatisticalModel = dataclasses.field(
      default_factory=network_models.NormalDistributionModel)
  mech_model: mechanistic_models.MechanisticModel = dataclasses.field(
      default_factory=mechanistic_models.ViboudChowellModel)
  fused_train_steps: int = 100

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

    epidemic_observables_fn = jax.vmap(mech_model.epidemic_observables)
    epidemic_observables = epidemic_observables_fn(mech_params, epidemics)
    statistical_log_likelihood = stat_model.log_likelihood(
        stat_params, covariates, epidemic_observables)

    mech_log_likelihood_fn = jax.vmap(mech_model.log_likelihood)
    mechanistic_log_likelihood = mech_log_likelihood_fn(mech_params, epidemics)
    return LogLikelihoods(
        stat_log_prior=statistical_log_prior,
        mech_log_prior=mechanistic_log_prior,
        mech_log_likelihood=mechanistic_log_likelihood,
        stat_log_likelihood=statistical_log_likelihood)

  def fit(self, data, train_steps, time_mask_value=30, seed=42):
    # TODO(dkochkov) consider a tunable module for preprocessing.
    data["total"] = (
        ("location", "time",), np.cumsum(
            data.new_infections.transpose("location", "time").values, -1))
    num_locations = data.sizes["location"]
    self.data = data
    self.covariates = covariates = data.static_covariates.transpose(
        "location", "static_covariate").values
    self.epidemics = epidemics = _pack_epidemics_record_tuple(data)
    self.time_mask = _get_time_mask(data, time_mask_value)

    # Mechanistic model initialization
    mech_params = jnp.stack([
        self.mech_model.init_parameters() for _ in range(num_locations)])
    observations = jax.vmap(self.mech_model.epidemic_observables)(
        mech_params, epidemics)

    # Statistical model initialization
    rng = jax.random.PRNGKey(seed)
    stat_params = self.stat_model.init_parameters(rng, covariates, observations)
    init_params = (stat_params, mech_params)

    opt_init, opt_update, get_params = optimizers.adam(5e-4, b1=0.9, b2=0.999)
    opt_state = opt_init(init_params)

    @jax.value_and_grad
    def negative_log_prob(params):
      log_likelihoods = self._log_likelihoods(
          params, epidemics, covariates, self.mech_model, self.stat_model)
      mech_log_likelihood = log_likelihoods.mech_log_likelihood
      mech_log_likelihood *= self.time_mask
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

  def predict(self, time_steps, num_samples, seed=0):
    rng = jax.random.PRNGKey(seed)
    if not hasattr(self, "params_"):
      raise AttributeError("`fit` must be called before `predict`.")
    # Should mech_params be sampled from a distribution instead?
    _, mech_params = self.params_
    predictions = monte_carlo.trajectories_from_model(
        self.mech_model, mech_params, rng,
        self.epidemics, time_steps, num_samples)

    # TODO(jamieas): consider indexing by seed.
    sample = np.arange(num_samples)

    # Here we assume evenly spaced integer time values.
    epidemic_time = self.data.time.data
    time_delta = epidemic_time[1] - epidemic_time[0]
    time = np.arange(1, time_steps + 1) * time_delta + epidemic_time[-1]

    location = self.data.location

    return xarray.DataArray(
        predictions,
        coords=[location, sample, time],
        dims=["location", "sample", "time"]).rename("new_infections")
