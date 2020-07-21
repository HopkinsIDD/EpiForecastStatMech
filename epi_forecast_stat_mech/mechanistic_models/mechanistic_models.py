# Lint as: python3
"""Mechanistic models for forecasting epidemics."""

import abc
import collections
from typing import Callable, Union, Mapping, Text
import dataclasses

from epi_forecast_stat_mech import utils

import jax
import jax.numpy as jnp
import numpy as np

import tensorflow_probability as tfp
tfp = tfp.experimental.substrates.jax
tfd = tfp.distributions

from epi_forecast_stat_mech.statistical_models import probability as stat_prob
from flax import nn
import flax.nn.initializers

Array = Union[float, jnp.DeviceArray, np.ndarray]


EpidemicsRecord = collections.namedtuple("EpidemicsRecord", [
    "t", "infections_over_time", "cumulative_infections", "dynamic_covariates"
])


def pack_epidemics_record_tuple(data):
  """Convert from xarray to the internal EpidemicsRecord."""
  # TODO(mcoram): Gaussian models probably need float32, but this straight
  #   conversion is problematic. E.g. datetime.datetime(2020, 5, 1) -> 1.6E18.
  tiled_time = np.tile(
      np.expand_dims(data.time.values.astype(np.float32), 0),
      (data.dims["location"], 1))
  # TODO(mcoram): Improve nan handling upstream and place guards.
  new_infections = data.new_infections.transpose("location", "time")
  dynamic_covariates = data.data_vars.get("dynamic_covariates", None)
  if dynamic_covariates is None:
    dynamic_covariates = jnp.zeros(
        (data.sizes["location"], data.sizes["time"], 0), dtype=np.float32)
  else:
    dynamic_covariates = dynamic_covariates.transpose(
        "location", "time", "dynamic_covariate").values.astype(np.float32)
  # TODO(mcoram): Improve nan handling upstream and place guards.
  new_infections_cumsum = new_infections.cumsum("time", skipna=True)
  return EpidemicsRecord(
      tiled_time,
      new_infections.values.astype(np.float32),
      new_infections_cumsum.values.astype(np.float32),
      dynamic_covariates)


@dataclasses.dataclass
class FastPoisson:
  """A Poisson distribution that uses a Normal approximation to sample for large rate."""

  rate: Array
  rate_split: float = 100.

  def sample(self, sample_shape, rng):
    """Draw samples from a Poisson distribution."""
    rng0, rng1 = jax.random.split(rng)
    use_poisson = self.rate < self.rate_split
    censored_rate = jnp.where(use_poisson, self.rate, 0.0)
    poisson_samples = tfd.Poisson(censored_rate).sample(
        sample_shape, seed=rng0)
    z = tfd.Normal(0.0, 1.0).sample(sample_shape, seed=rng1)
    normal_floats = (0.5 * z + jnp.sqrt(self.rate)) ** 2
    normal_ints = jnp.maximum(normal_floats.round().astype(jnp.int32), 0)
    return jnp.where(use_poisson, poisson_samples, normal_ints)

  def log_prob(self, x, **kwargs):
    # The use of where in the following is a workaround for the "double-where"
    # issue in which nan's propogate through vjp in unexpected ways.
    # c.f. https://github.com/google/jax/issues/1052
    return tfd.Poisson(self.rate).log_prob(
        jnp.where(x >= 0, x, -1.), **kwargs)


def OverDispersedPoisson(mean, overdispersion):
  """OverDispersedPoisson distribution.

  Has mean: mean
  Has variance: overdispersion * mean

  Arguments:
    mean: tensor of non-negative floats.
    overdispersion: tensor of floats > 1 (strictly!).
  Returns:
    tfp.Distribution
  """
  # TODO(mcoram): Address the double-where here too.
  return tfd.NegativeBinomial(
      total_count=mean / (overdispersion - 1),
      probs=(1. - 1. / overdispersion),
      validate_args=False)


class MechanisticModel:
  """Abstract class representing mechanistic models."""

  @abc.abstractmethod
  def log_prior(self, parameters):
    """Returns the log prior probability of `parameters`.

    Args:
      parameters: parameters of the mechanistic model.

    Returns:
      log-probabilities for each parameter. Must be of the same structure as
      parameters.
    """
    ...

  @abc.abstractmethod
  def log_likelihood(self, parameters, epidemics):
    """Returns the log likelihood of `epidemics` given `parameters`.

    Args:
      parameters: parameters of the mechanistic model. Can be arbitrary pytree.
      epidemics: array representing number of infections as a function of time.

    Returns:
      array of the same shape as `epidemics` containing log-likelihoods of the
      observed data under the model with parameters `parameters`.
    """
    ...

  @abc.abstractmethod
  def predict(
      self,
      parameters,
      rng,
      observed_epidemics,
      length,
      include_observed=True
  ):
    """Samples a trajectory continuing `observed_epidemics` of lenth `length.

    Args:
      parameters: parameters of the mechanistic model.
      rng: random number generator of type `jax.random.PRNG`.
      observed_epidemics: array representing number of infections as a function
        of time at the observed time course of the epidemics.
      length: number of time intervals for which to forecast the epidemics.
      include_observed: whether to include the `observed_epidemics` in the
        output.

    Returns:
      trajectory of predicted infections of length `length` or
      `len(observed_epidemics) + length` if `include_observed` is True.
    """
    ...

  @abc.abstractmethod
  def init_parameters(self):
    """Returns reasonable `parameters` for an initial guess."""
    ...

  @property
  @abc.abstractmethod
  def encoded_param_names(self):
    ...

  @property
  @abc.abstractmethod
  def param_names(self):
    ...

  @property
  @abc.abstractmethod
  def bottom_scale(self):
    ...

  def decode_params(self, parameters):
    # This won't actually work when split_and_scale_parameters is not defined.
    # A more "grown-up" version requires flattening anyway.
    return jnp.concatenate(
        self.split_and_scale_parameters(parameters), axis=-1)

  @abc.abstractmethod
  def encode_params(self, parameters):
    ...


# TODO(dkochkov) Consider adding rng to the state transition method.
@dataclasses.dataclass
class IntensityModel(MechanisticModel):
  """Base intensity model class.

  This class of mechanistic models evaluates log_likelihood of the epidemic
  trajectory sequentially, modeling the number of new cases as samples from a
  specified probability distribution `self.new_infection_distribution`. The
  `self._intensity` method, dependent on a hidden state of the model,
  parametrizes the `self.new_infection_distribution`.
  """

  # A callable that takes a single argument `mean` and returns an object
  # that has a `.predict` and `log_prob` method. Typically, this is a TensorFlow
  # probability distribution.
  new_infection_distribution: Callable = FastPoisson  # pylint: disable=g-bare-generic

  @abc.abstractmethod
  def _intensity(self, parameters, state):
    """Computes intensity given `parameters` and `state`."""
    ...

  @abc.abstractmethod
  def _update_state(self, parameters, state, new_cases):
    """Computes an update to the internal state of the model."""
    ...

  @abc.abstractmethod
  def _initial_state(self, parameters, initial_record):
    """Initializes the hidden state of the model based on initial data."""
    ...

  def log_likelihood(self, parameters, epidemic_record):
    """Returns the log likelihood of the `epidemic_record` under the model."""
    def _log_prob_step(state, new_cases_and_params):
      new_cases, current_params = new_cases_and_params
      intensity = self._intensity(current_params, state)
      log_prob = self.new_infection_distribution(*intensity).log_prob(new_cases)
      new_state = self._update_state(current_params, state, new_cases)
      return new_state, jnp.squeeze(log_prob)

    time_series_length = epidemic_record.t.shape[0]
    time_dependent_params_shape = (time_series_length,
                                   len(self.step_based_encoded_param_names))
    time_dependent_params = jax.tree_multimap(jnp.broadcast_to, parameters,
                                              time_dependent_params_shape)
    start_params, params = utils.split_along_axis(time_dependent_params, 0, 1)
    start_record, record = utils.split_along_axis(epidemic_record, 0, 1)
    start_state = self._initial_state(start_params, start_record)
    new_cases_and_params = (record.infections_over_time, params)
    _, log_probs = jax.lax.scan(
        _log_prob_step, start_state, new_cases_and_params)
    # we include 0 to indicate that we treat the first step as given.
    return jnp.concatenate([jnp.zeros(1), log_probs])

  def predict(
      self,
      parameters,
      rng,
      observed_epidemic_record,
      length,
      include_observed=True
  ):
    """Samples a trajectory continuing `observed_epidemic` by length `length`.

    Args:
      parameters: parameters of the mechanistic model.
      rng: PRNG to use for sampling.
      observed_epidemic_record: initial epidemics trajectory to continue.
      length: how many steps to unroll epidemics for.
      include_observed: whether to prepend the observed epidemics.

    Returns:
      trajectory of predicted infections of length `length` or
      `len(observed_epidemic_record) + length` if `include_observed` is True.
    """
    def _transition(state_and_rng, new_cases_and_params):
      state, rng = state_and_rng
      new_cases, params = new_cases_and_params
      if new_cases is not None:
        new_state = self._update_state(params, state, new_cases)
      else:
        current_rng, rng = jax.random.split(rng)
        intensity = self._intensity(params, state)
        new_cases = jnp.squeeze(
            self.new_infection_distribution(*intensity).sample(1, current_rng))
        new_state = self._update_state(params, state, new_cases)
      return (new_state, rng), new_cases

    observed_record_length = observed_epidemic_record.t.shape[0]
    time_series_length = observed_record_length + length
    time_dependent_params_shape = (time_series_length,
                                   len(self.step_based_encoded_param_names))
    time_dependent_params = jax.tree_multimap(jnp.broadcast_to, parameters,
                                              time_dependent_params_shape)
    start_params, params = utils.split_along_axis(time_dependent_params, 0, 1)
    start_record, record = utils.split_along_axis(
        observed_epidemic_record, 0, 1)
    params_for_record, params_for_prediction = utils.split_along_axis(
        params, 0, observed_record_length - 1, post_squeeze_first=False)

    start_state = self._initial_state(start_params, start_record)
    state_and_rng = (start_state, rng)
    # TODO(mcoram): Fix the state part. And make a second for unroll.
    new_cases_and_params = (record.infections_over_time, params_for_record)
    state_and_rng, _ = jax.lax.scan(
        _transition, state_and_rng, new_cases_and_params)
    new_cases_and_params = (None, params_for_prediction)
    _, unroll = jax.lax.scan(_transition, state_and_rng, new_cases_and_params)
    if include_observed:
      observed_cases = observed_epidemic_record.infections_over_time
      return jnp.concatenate([observed_cases, unroll])
    return unroll


class StepBasedViboudChowellModel(IntensityModel):
  """ViboudChowell mechanistic model."""

  @property
  def step_based_param_names(self):
    return ("r", "a", "p", "K")

  def encode_params(self, parameters):
    return jnp.log(parameters)

  @property
  def step_based_encoded_param_names(self):
    return ("log_r", "log_a", "log_p", "log_K")

  @property
  def bottom_scale(self):
    return jnp.asarray((.1, .1, .1, .1))

  def _intensity(self, parameters, state):
    """Computes intensity given `parameters`, `state`."""
    r, a, p, k = self._split_and_scale_parameters(parameters)
    x = state
    return (jnp.maximum(r * x ** p * jnp.maximum(1 - (x / k), 1E-6) ** a, 0.1),)

  def _update_state(self, parameters, state, new_cases):
    """Computes an update to the internal state of the model."""
    del parameters  # unused
    # Skip state update if new_cases is nan (if-based implemention breaks jit).
    return jnp.where(
        jnp.isnan(new_cases),
        state,
        state + new_cases)

  def _initial_state(self, parameters, initial_record):
    """Initializes the hidden state of the model based on initial data."""
    del parameters  # unused
    return jnp.where(
        jnp.isnan(initial_record.cumulative_infections), 0.,
        initial_record.cumulative_infections)

  def init_parameters(self):
    """Returns reasonable `parameters` for an initial guess."""
    return self.encode_params(jnp.asarray([2., .9, .9, 200000.]))

  def _split_and_scale_parameters(self, parameters):
    """Splits parameters and scales them appropriately.

    We use convention in which parameters of a ViboudChowell model are stored
    in an array as: `np.array([log(r), log(a), log(p), log(k)])`.

    Args:
      parameters: array containing parametrization of the model.

    Returns:
      (r, a, p, k) parameters.
    """
    return jnp.split(jnp.exp(parameters), 4, axis=-1)


class StepBasedGaussianModel(IntensityModel):
  """Gaussian mechanistic model."""

  def _intensity(self, parameters, state):
    """Computes intensity at time `t` given `parameters`."""
    m, s, k = self._split_and_scale_parameters(parameters)
    t = state
    dist = tfd.Normal(loc=m, scale=s)
    return jnp.maximum(k * (dist.cdf(t) - dist.cdf(t - 1.)), 0.1)

  def _update_state(self, parameters, state, new_cases):
    """Computes an update to the internal state of the model."""
    del parameters, new_cases  # unused
    # We choose to advance time whether or not the current new_cases is nan.
    return state + 1.

  def _initial_state(self, parameters, initial_record):
    """Initializes the hidden state of the model based on initial data."""
    del parameters  # unused
    return initial_record.t

  def _split_and_scale_parameters(self, parameters):
    """Splits parameters and scales them appropriately.

    We use convention in which parameters of a Gaussian model are stored
    in an array as: `np.array([m, log(s), log(k)])`.

    Args:
      parameters: array containing parametrization of the model.

    Returns:
      (m, s, k) parameters.
    """
    m, log_s, log_k = jnp.split(parameters, 3, axis=-1)
    return m, jnp.exp(log_s), jnp.exp(log_k)

  def encode_params(self, parameters):
    return jnp.concatenate((parameters[[0]], jnp.log(parameters[1:])), axis=-1)

  def init_parameters(self):
    """Returns reasonable `parameters` for an initial guess."""
    return self.encode_params(jnp.asarray([100., 100., 1000.]))

  @property
  def step_based_param_names(self):
    return ("m", "s", "K")

  @property
  def step_based_encoded_param_names(self):
    return ("m", "log_s", "log_K")


class StepBasedMultiplicativeGrowthModel(IntensityModel):
  """MultiplicativeGrowth mechanistic model."""

  new_infection_distribution: Callable = OverDispersedPoisson

  @property
  def step_based_param_names(self):
    return ("base", "beta", "K")

  def encode_params(self, parameters):
    return jnp.log(parameters)

  @property
  def step_based_encoded_param_names(self):
    return ("log_base", "log_beta", "log_K")

  @property
  def bottom_scale(self):
    return jnp.asarray((.1, .1, .1))

  def _split_and_scale_parameters(self, parameters):
    """Splits parameters and scales them appropriately.


    Args:
      parameters: array containing parametrization of the model.

    Returns:
      (base, beta, K) parameters.
    """
    tuple_form = (base, beta, K) = jnp.exp(parameters)
    return tuple_form

  def init_parameters(self):
    """Returns reasonable `parameters` for an initial guess."""
    return self.encode_params(jnp.asarray([0.5, 0.75, 10000.]))

  def _initial_state(self, parameters, initial_record):
    """Initializes the hidden state of the model based on initial data."""
    del parameters  # unused
    return (jnp.maximum(jnp.nan_to_num(initial_record.infections_over_time), 1),
            jnp.nan_to_num(initial_record.cumulative_infections))

  def _update_state(self, parameters, state, o):
    """Computes an update to the internal state of the model."""
    o_hat, unused_overdispersion = self._intensity(parameters, state)
    old_o_smooth, old_cumulative_cases = state
    base, beta, K = self._split_and_scale_parameters(parameters)
    # Improve this: e.g. it should pay more attention to o when o is big.
    eta = 0.25
    o_smooth = eta * o + (1. - eta) * o_hat
    cumulative_cases = old_cumulative_cases + o
    # Skip the update if new_cases == o is np.nan.
    final_o_smooth = jnp.where(
        jnp.isnan(o),
        old_o_smooth,
        o_smooth)
    final_cumulative_cases = jnp.where(
        jnp.isnan(o),
        old_cumulative_cases,
        cumulative_cases)
    return final_o_smooth, final_cumulative_cases

  def _intensity(self, parameters, state):
    """Computes intensity given `parameters`, `state`."""
    base, beta, K = self._split_and_scale_parameters(parameters)
    old_o_smooth, cumulative_cases = state
    multiplier = base + beta * jnp.maximum(0., (1. - cumulative_cases / K))
    o_hat = multiplier * old_o_smooth
    overdispersion = 2.
    return (jnp.maximum(o_hat, 0.1), overdispersion,)

  # These are kind of a compatibility layer thing.
  @property
  def param_names(self):
    return self.step_based_param_names

  @property
  def encoded_param_names(self):
    return self.step_based_encoded_param_names

  def decode_params(self, parameters):
    return jnp.exp(jnp.asarray(parameters))

  def log_prior(self, parameters):
    """Returns log_probability prior of the `parameters` of the model."""
    return jnp.zeros_like(parameters)




# TODO(jamieas): unify VC and Gaussian models as subclasses of `IntensityModel`.
# TODO(dkochkov) add pytype annotations, refer to decided `EpidemicsRecord`.
# TODO(dkochkov) consider moving paramteres to flax params.
# TODO(dkochkov) consider combining VC and Gaussian models in IntensityFamily
@dataclasses.dataclass
class ViboudChowellModel(MechanisticModel):
  """ViboudChowell mechanistic model."""

  # A callable that takes a single argument `mean` and returns an object
  # that has a `.predict` and `log_prob` method. Typically, this is a TensorFlow
  # Probability distribution.
  new_infection_distribution: Callable = FastPoisson  # pylint: disable=g-bare-generic

  def log_prior(self, parameters):
    r, a, p, k = self.split_and_scale_parameters(parameters)
    del r, k
    # TODO(dkochkov) use more informed priors, this currently works ok.
    a_prior = tfd.Normal(1, 0.5).log_prob(a)
    p_prior = tfd.Normal(1, 0.5).log_prob(p)
    return jnp.stack([a_prior, p_prior], axis=-1)

  def intensity(self, parameters, x):
    """Computes intensity given `parameters` and number of current_cases `x`."""
    r, a, p, k = self.split_and_scale_parameters(parameters)
    return jnp.maximum(r * x ** p * (jnp.maximum(1 - x / k, 1E-6)) ** a, 0.1)

  def split_and_scale_parameters(self, parameters):
    """Splits parameters and scales them appropriately.

    We use convention in which parameters of a ViboudChowell model are stored
    in an array as: `np.array([log(r), log(a), log(p), log(k)])`.

    Args:
      parameters: array containing parametrization of the model.

    Returns:
      (r, a, p, k) parameters.
    """
    return jnp.split(jnp.exp(parameters), 4, axis=-1)

  @property
  def param_names(self):
    return ("r", "a", "p", "K")

  def encode_params(self, parameters):
    return jnp.log(parameters)

  @property
  def encoded_param_names(self):
    return ("log_r", "log_a", "log_p", "log_K")

  @property
  def bottom_scale(self):
    return jnp.asarray((.1, .1, .1, .1))

  def init_parameters(self):
    """Returns reasonable `parameters` for an initial guess."""
    return self.encode_params(jnp.asarray([2., .9, .9, 250000.]))

  def log_likelihood(self, parameters, epidemics):
    """Returns the log likelihood of `epidemics` given `parameters`."""
    intensity = self.intensity(parameters, epidemics.cumulative_infections[:-1])
    trajectory = epidemics.infections_over_time[1:]
    log_probs = self.new_infection_distribution(intensity).log_prob(trajectory)
    return jnp.concatenate([jnp.zeros(1), log_probs])

  def predict(
      self,
      parameters,
      rng,
      observed_epidemics,
      length,
      include_observed=True
  ):
    """Samples a trajectory continuing `observed_epidemics` by length `length`.

    Args:
      parameters: parameters of the mechanistic model.
      rng: PRNG to use for sampling.
      observed_epidemics: initial epidemics trajectory to continue.
      length: how many steps to unroll epidemics for.
      include_observed: whether to prepend the observed epidemics.

    Returns:
      trajectory of predicted infections of length `length` or
      `len(observed_epidemics) + length` if `include_observed` is True.
    """
    cumulative_cases = observed_epidemics.cumulative_infections[-1]
    def _step(rng_and_cumulative, _):
      rng, cumulative_cases = rng_and_cumulative
      next_rng, rng = jax.random.split(rng)
      intensity = self.intensity(parameters, cumulative_cases)
      new_cases = jnp.squeeze(
          self.new_infection_distribution(intensity).sample(1, rng))
      cumulative_cases += new_cases
      return (next_rng, cumulative_cases), new_cases

    _, unroll = jax.lax.scan(_step, (rng, cumulative_cases), None, length)
    if include_observed:
      return jnp.concatenate([observed_epidemics.infections_over_time, unroll])
    return unroll


@dataclasses.dataclass
class GaussianModel(MechanisticModel):
  """Mechanistic model based on gaussian shape of the epidemics."""

  # A callable that takes a single argument `mean` and returns an object
  # that has a `.predict` and `log_prob` method. Typically, this is a TensorFlow
  # Probability distribution.
  new_infection_distribution: Callable = FastPoisson  # pylint: disable=g-bare-generic

  def log_prior(self, parameters):
    """Returns log_probability prior of the `parameters` of the model."""
    # TODO(dkochkov) consider adding real prior to the model.
    return jnp.zeros_like(parameters)

  def intensity(self, parameters, t):
    """Computes intensity at time `t` given `parameters`."""
    m, s, k = self.split_and_scale_parameters(parameters)
    dist = tfd.Normal(loc=m, scale=s)
    return jnp.maximum(k * (dist.cdf(t) - dist.cdf(t - 1.)), 0.1)

  def split_and_scale_parameters(self, parameters):
    """Splits parameters and scales them appropriately.

    We use convention in which parameters of a Gaussian model are stored
    in an array as: `np.array([m, log(s), log(k)])`.

    Args:
      parameters: array containing parametrization of the model.

    Returns:
      (m, s, k) parameters.
    """
    m, log_s, log_k = jnp.split(parameters, 3, axis=-1)
    return m, jnp.exp(log_s), jnp.exp(log_k)

  @property
  def param_names(self):
    return ("m", "s", "K")

  def encode_params(self, parameters):
    return jnp.concatenate((parameters[[0]], jnp.log(parameters[1:])), axis=-1)

  @property
  def encoded_param_names(self):
    return ("m", "log_s", "log_K")

  @property
  def bottom_scale(self):
    return jnp.asarray((3., .1, .1))

  def init_parameters(self):
    """Returns reasonable `parameters` for an initial guess."""
    return self.encode_params(jnp.asarray([100., 100., 1000.]))

  def log_likelihood(self, parameters, epidemics):
    """Returns the log likelihood of `epidemics` given `parameters`."""
    intensity = self.intensity(parameters, epidemics.t)
    trajectory = epidemics.infections_over_time
    log_probs = self.new_infection_distribution(intensity).log_prob(trajectory)
    return log_probs

  def predict(
      self,
      parameters,
      rng,
      observed_epidemics,
      length,
      include_observed=True
  ):
    """Samples a trajectory continuing `observed_epidemics` by lenth `length`.

    Args:
      parameters: parameters of the mechanistic model.
      rng: PRNG to use for sampling.
      observed_epidemics: a namedtuple containing information about the initial
        trajectory of the epidemics. Must contain fields `t` filed for time and
        `infections_over_time` for number infections on corresponding day.
      length: how many steps to unroll epidemics for.
      include_observed: whether to prepend the observed epidemics.

    Returns:
      trajectory of predicted infections of length `length` or
      `len(observed_epidemics) + length` if `include_observed` is True.
    """
    start_time = observed_epidemics.t[-1] + 1.  # first time to predict.
    def _step(rng_and_time, _):
      rng, t = rng_and_time
      next_rng, rng = jax.random.split(rng)
      intensity = self.intensity(parameters, t)
      new_cases = jnp.squeeze(
          self.new_infection_distribution(intensity).sample(1, rng))
      return (next_rng, t + 1), new_cases

    _, unroll = jax.lax.scan(_step, (rng, start_time), None, length)
    if include_observed:
      return jnp.concatenate([observed_epidemics.infections_over_time, unroll])
    return unroll


@dataclasses.dataclass
class ViboudChowellModelPseudoLikelihood(ViboudChowellModel):
  """ViboudChowell mechanistic model with a pseudo-likelihood criterion."""

  def log_likelihood(self, parameters, epidemics):
    """Returns the (pseudo) log likelihood of `epidemics` given `parameters`."""
    # We treat the calculations as if they are conditional on observation[0]
    # for consistency with the other VC code. So the intensity and observations
    # vectors are one short, and the shortfall is filled in with 0. at the end.
    intensity = self.intensity(parameters, epidemics.cumulative_infections[:-1])
    observations = epidemics.infections_over_time[1:]
    log_probs = stat_prob.pseudo_poisson_log_probs(intensity, observations)
    return jnp.concatenate([jnp.zeros(1), log_probs])


@dataclasses.dataclass
class GaussianModelPseudoLikelihood(GaussianModel):
  """Mechanistic model of Gaussian shape with a pseudo-likelihood criterion."""

  def log_likelihood(self, parameters, epidemics):
    """Returns the (pseudo) log likelihood of `epidemics` given `parameters`."""
    intensity = self.intensity(parameters, epidemics.t)
    observations = epidemics.infections_over_time
    log_probs = stat_prob.pseudo_poisson_log_probs(intensity, observations)
    return log_probs


def constant_initializer(value):

  def init(key, shape, dtype=jnp.float32):
    return jnp.asarray(value).reshape(shape).astype(dtype)

  return init


class ConstantModule(nn.Module):

  def apply(self, x, init):
    bias = self.param("bias", init.shape, constant_initializer(init))
    return jnp.broadcast_to(bias, x.shape[:-1] + bias.shape)


def params_dict_path_to_leaves(params, dynamic_covariate, path=()):
  """Replace a params "tree"'s leaves with named leaves."""
  out = {}
  if set(["bias", "kernel"]).issuperset(set(params.keys())):
    bias = params.get("bias", None)
    kernel = params.get("kernel", None)
    if bias is not None:
      out["bias"] = np.asarray([
          "__".join(path),
      ])
    if kernel is not None:
      out["kernel"] = np.asarray([
          "__".join(path + (f"from_{key}",)) for key in dynamic_covariate.data
      ])
    return out
  for key, val in params.items():
    out[key] = params_dict_path_to_leaves(val, dynamic_covariate, path + (key,))
  return out


def params_dict_kernel_indicator(params):
  """Replace a params "tree"'s leaves with indicator of kernel."""
  out = {}
  if set(["bias", "kernel"]).issuperset(set(params.keys())):
    bias = params.get("bias", None)
    kernel = params.get("kernel", None)
    if bias is not None:
      out["bias"] = jnp.zeros_like(bias)
    if kernel is not None:
      out["kernel"] = jnp.ones_like(kernel)
    return out
  for key, val in params.items():
    out[key] = params_dict_kernel_indicator(val)
  return out


class DynamicIntensityModel(IntensityModel):
  """Wrap an IntensityModel with a DynamicModule."""

  def __init__(self, rng, dynamic_covariate):
    self.dynamic_covariate = dynamic_covariate
    DynamicModule = self.DynamicModule
    _, init_params = DynamicModule.init_by_shape(
        rng, [((len(dynamic_covariate),), jnp.float32)])
    self.init_params = init_params
    init_flat, unravel = jax.flatten_util.ravel_pytree(init_params)
    self.unravel = unravel
    self.init_flat = init_flat
    kernel_indicator = params_dict_kernel_indicator(init_params)
    flat_kernel_indicator, _ = jax.flatten_util.ravel_pytree(kernel_indicator)
    self.flat_kernel_indicator = flat_kernel_indicator
    super().__init__()

  @property
  def encoded_param_names(self):
    return tuple(
        np.concatenate(
            jax.tree_flatten(
                params_dict_path_to_leaves(
                    self.init_params, self.dynamic_covariate))[0]))

  @property
  def param_names(self):
    return self.encoded_param_names

  def encode_params(self, flat_params):
    return flat_params

  def split_and_scale_parameters(self, flat_params):
    return jnp.split(flat_params, len(flat_params))

  def decode_params(self, flat_params):
    return flat_params

  @property
  def bottom_scale(self):
    return jnp.asarray(
        [self.bottom_scale_dict[key] for key in self.encoded_param_names])

  def init_parameters(self):
    return self.init_flat

  def log_prior(self, parameters):
    """Returns log_probability prior of the `parameters` of the model."""
    return jnp.zeros_like(parameters)

  def log_likelihood(self, flat_parameters, epidemic_record):
    parameters = self.unravel(flat_parameters)
    time_dep_params = self.DynamicModule.call(
        parameters, epidemic_record.dynamic_covariates)
    return super().log_likelihood(time_dep_params, epidemic_record)

  def predict(
      self,
      flat_parameters,
      rng,
      observed_epidemic_record,
      dynamic_covariates,
      include_observed=True
  ):
    """Samples a trajectory continuing `observed_epidemic` by length `length`.

    Args:
      flat_parameters: parameters of the mechanistic model.
      rng: PRNG to use for sampling.
      observed_epidemic_record: initial epidemics trajectory to continue.
      dynamic_covariates: np array of time x dynamic_covariate.
      include_observed: whether to prepend the observed epidemics.

    Returns:
      trajectory of predicted infections of length `length` or
      `len(observed_epidemic_record) + length` if `include_observed` is True.
    """
    parameters = self.unravel(flat_parameters)
    time_dep_params = self.DynamicModule.call(
        parameters, dynamic_covariates)
    length = dynamic_covariates.shape[0] - len(observed_epidemic_record.t)
    assert length >= 0
    return super().predict(time_dep_params, rng, observed_epidemic_record,
                           length, include_observed)


class DynamicMultiplicativeGrowthModel(DynamicIntensityModel,
                                       StepBasedMultiplicativeGrowthModel):
  """MultiplicativeGrowth mechanistic model with dynamic variable utilization."""

  def __init__(self, rng, dynamic_covariate):
    self.bottom_scale_dict = collections.defaultdict(lambda: 0.1)
    super().__init__(rng, dynamic_covariate)

  class DynamicModule(nn.Module):

    def apply(self, x):
      log_base_module = ConstantModule.partial(
          name="log_base",
          init=jnp.reshape(jnp.log(0.5), (1,)))
      log_beta_module = nn.Dense.partial(
          name="log_beta",
          features=1,
          kernel_init=flax.nn.initializers.zeros,
          bias_init=constant_initializer(jnp.log(0.75)))
      log_K_module = ConstantModule.partial(
          name="log_K",
          init=jnp.reshape(jnp.log(10000.), (1,)))
      log_base = log_base_module(x)
      log_beta = log_beta_module(x)
      log_K = log_K_module(x)
      return jnp.concatenate((log_base, log_beta, log_K), axis=-1)


def discrete_seir_update(susceptible, exposed, infected, exposure_rate,
                         symptom_rate, recovery_rate):
  """All quantities are in per-population units."""
  new_exposures = exposure_rate * susceptible * infected
  new_infections = symptom_rate * exposed
  new_recoveries = recovery_rate * infected
  return (susceptible - new_exposures, exposed + new_exposures - new_infections,
          infected + new_infections - new_recoveries)


class StepBasedBaselineSEIRModel(IntensityModel):
  """Baseline SEIR mechanistic model."""

  new_infection_distribution: Callable = FastPoisson

  @property
  def step_based_param_names(self):
    return ("exposure_rate", "symptom_rate", "recovery_rate", "K")

  def encode_params(self, parameters):
    return jnp.log(parameters)

  @property
  def step_based_encoded_param_names(self):
    return ("log_exposure_rate", "log_symptom_rate", "log_recovery_rate",
            "log_K")

  @property
  def bottom_scale(self):
    return jnp.asarray((.1, .1, .1, .1))

  def _split_and_scale_parameters(self, parameters):
    """Splits parameters and scales them appropriately.


    Args:
      parameters: array containing parametrization of the model.

    Returns:
      (exposure_rate, symptom_rate, recovery_rate, K) parameters.
    """
    tuple_form = (exposure_rate, symptom_rate, recovery_rate,
                  K) = jnp.exp(parameters)
    return tuple_form

  def init_parameters(self):
    """Returns reasonable `parameters` for an initial guess."""
    return self.encode_params(jnp.asarray([0.5, 0.3, 0.3, 10000.]))

  def _initial_state(self, parameters, initial_record):
    """Initializes the hidden state of the model based on initial data."""
    del parameters  # unused
    # Hidden state is (susceptible, exposed, infected) in per-population units.
    # TODO(mcoram): Consider making initial log_infected_frac a parameter.
    return (0.999, 0., 0.001)

  def _update_state(self, parameters, state, unused_new_infections):
    """Computes an update to the internal state of the model."""
    susceptible, exposed, infected = state
    exposure_rate, symptom_rate, recovery_rate, K = self._split_and_scale_parameters(
        parameters)
    new_susceptible, new_exposed, new_infected = discrete_seir_update(
        susceptible, exposed, infected, exposure_rate, symptom_rate,
        recovery_rate)
    return new_susceptible, new_exposed, new_infected

  def _intensity(self, parameters, state):
    """Computes intensity given `parameters`, `state`."""
    unused_susceptible, exposed, unused_infected = state
    (unused_exposure_rate, symptom_rate, unused_recovery_rate,
     K) = self._split_and_scale_parameters(parameters)
    # Observe that K folds together population and observation_rate.
    # Return a singleton tuple holding a float for the Poisson parameter.
    return (jnp.maximum(K * symptom_rate * exposed, 0.1),)

  # These are kind of a compatibility layer thing.
  @property
  def param_names(self):
    return self.step_based_param_names

  @property
  def encoded_param_names(self):
    return self.step_based_encoded_param_names

  def decode_params(self, parameters):
    return jnp.exp(jnp.asarray(parameters))

  def log_prior(self, parameters):
    """Returns log_probability prior of the `parameters` of the model."""
    return jnp.zeros_like(parameters)


class DynamicBaselineSEIRModel(DynamicIntensityModel,
                               StepBasedBaselineSEIRModel):
  """BaselineSEIR mechanistic model with dynamic variable utilization."""

  def __init__(self, rng, dynamic_covariate):
    self.bottom_scale_dict = collections.defaultdict(lambda: 0.1)
    super().__init__(rng, dynamic_covariate)

  class DynamicModule(nn.Module):

    def apply(self, x):
      # exposure_rate, symptom_rate, recovery_rate, K
      log_exposure_rate_module = ConstantModule.partial(
          name="log_exposure_rate", init=jnp.reshape(jnp.log(0.5), (1,)))
      log_symptom_rate_module = nn.Dense.partial(
          name="log_symptom_rate",
          features=1,
          kernel_init=flax.nn.initializers.zeros,
          bias_init=constant_initializer(jnp.log(0.3)))
      log_recovery_rate_module = ConstantModule.partial(
          name="log_recovery_rate", init=jnp.reshape(jnp.log(0.3), (1,)))
      log_K_module = ConstantModule.partial(
          name="log_K", init=jnp.reshape(jnp.log(10000.), (1,)))
      log_exposure_rate = log_exposure_rate_module(x)
      log_symptom_rate = log_symptom_rate_module(x)
      log_recovery_rate = log_recovery_rate_module(x)
      log_K = log_K_module(x)
      return jnp.concatenate(
          (log_exposure_rate, log_symptom_rate, log_recovery_rate, log_K),
          axis=-1)
