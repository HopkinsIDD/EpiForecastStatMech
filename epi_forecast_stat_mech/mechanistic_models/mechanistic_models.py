# Lint as: python3
"""Mechanistic models for forecasting epidemics."""

import abc
import collections
from typing import Callable, Union
import dataclasses

from epi_forecast_stat_mech import utils

import jax
import jax.numpy as jnp
import numpy as np

import tensorflow_probability as tfp
tfp = tfp.experimental.substrates.jax
tfd = tfp.distributions

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

  def log_prob(self, *args, **kwargs):
    return tfd.Poisson(self.rate).log_prob(*args, **kwargs)


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
  def epidemic_observables(self, parameters, epidemics):
    """Computes statistical observables of the epidemics.

    Could be any set of values that represent global properties of the epidemics
    e.g. total size of the epidemics or peak location.

    Args:
      parameters: parameters of the mechanistic model.
      epidemics: observed epidemic trajectory.

    Returns:
      Pytree holding the estimates of statistical properties of
      the epidemic based on the mechanistic model. This can vary between all
      model parameters to a subset or other statistical predictions by the
      model.
    """
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
    time_dependent_params_shape = jax.tree_map(
        lambda x: (time_series_length,) + x.shape, self.init_parameters())
    time_dependent_params = jax.tree_multimap(
        jnp.broadcast_to, parameters, time_dependent_params_shape)
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
        new_state = self._update_state(parameters, state, new_cases)
      else:
        current_rng, rng = jax.random.split(rng)
        intensity = self._intensity(params, state)
        new_cases = jnp.squeeze(
            self.new_infection_distribution(*intensity).sample(1, current_rng))
        new_state = self._update_state(parameters, state, new_cases)
      return (new_state, rng), new_cases

    observed_record_length = observed_epidemic_record.t.shape[0]
    time_series_length = observed_record_length + length
    time_dependent_params_shape = jax.tree_map(
        lambda x: (time_series_length,) + x.shape, self.init_parameters())
    time_dependent_params = jax.tree_multimap(
        jnp.broadcast_to, parameters, time_dependent_params_shape)
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

  def _intensity(self, parameters, state):
    """Computes intensity given `parameters`, `state`."""
    r, a, p, k = self._split_and_scale_parameters(parameters)
    x = state
    return (jnp.maximum(r * x ** p * jnp.maximum(1 - (x / k), 1E-6) ** a, 0.1),)

  def _update_state(self, parameters, state, new_cases):
    """Computes an update to the internal state of the model."""
    del parameters  # unused
    return state + new_cases

  def _initial_state(self, parameters, initial_record):
    """Initializes the hidden state of the model based on initial data."""
    del parameters  # unused
    return initial_record.cumulative_infections

  @staticmethod
  def init_parameters():
    """Returns reasonable `parameters` for an initial guess."""
    return jnp.log(jnp.asarray([2., .9, .9, 200000.]))

  def epidemic_observables(self, parameters, epidemics):
    """See base class."""
    return {"epidemic_size": parameters[..., -1:]}

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

  @staticmethod
  def init_parameters():
    """Returns reasonable `parameters` for an initial guess."""
    return jnp.asarray([100., np.log(100.), np.log(1000.)])


class StepBasedMultiplicativeGrowth(IntensityModel):
  """MultiplicativeGrowth mechanistic model."""

  new_infection_distribution: Callable = OverDispersedPoisson

  @property
  def param_names(self):
    return ("base", "beta", "K")

  def encode_params(self, parameters):
    return jnp.log(parameters)

  @property
  def encoded_param_names(self):
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
    return jnp.split(jnp.exp(parameters), 3, axis=-1)

  @staticmethod
  def init_parameters():
    """Returns reasonable `parameters` for an initial guess."""
    return jnp.log(jnp.asarray([0.5, 0.75, 10000.]))

  def _initial_state(self, parameters, initial_record):
    """Initializes the hidden state of the model based on initial data."""
    del parameters  # unused
    return (jnp.maximum(initial_record.infections_over_time,
                        1), initial_record.cumulative_infections)

  def _update_state(self, parameters, state, o):
    """Computes an update to the internal state of the model."""
    o_hat, unused_overdispersion = self._intensity(parameters, state)
    unused_old_o_smooth, old_cumulative_cases = state
    base, beta, K = self._split_and_scale_parameters(parameters)
    # Improve this: e.g. it should pay more attention to o when o is big.
    eta = 0.25
    o_smooth = eta * o + (1. - eta) * o_hat
    cumulative_cases = old_cumulative_cases + o
    return o_smooth, cumulative_cases

  def _intensity(self, parameters, state):
    """Computes intensity given `parameters`, `state`."""
    base, beta, K = self._split_and_scale_parameters(parameters)
    old_o_smooth, cumulative_cases = state
    multiplier = jnp.squeeze(base + beta * jnp.maximum(0., (1. - cumulative_cases / K)))
    o_hat = multiplier * old_o_smooth
    overdispersion = 2.
    return (jnp.maximum(o_hat, 0.1), overdispersion,)

  def epidemic_observables(self, parameters, epidemics):
    """See base class."""
    return {"epidemic_size": parameters[..., -1:]}



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

  @staticmethod
  def init_parameters():
    """Returns reasonable `parameters` for an initial guess."""
    return jnp.log(jnp.asarray([2., .9, .9, 250000.]))

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

  def epidemic_observables(self, parameters, epidemics):
    """See base class."""
    return {"epidemic_size": parameters[..., -1:]}


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

  @staticmethod
  def init_parameters():
    """Returns reasonable `parameters` for an initial guess."""
    return jnp.asarray([100., np.log(100.), np.log(1000.)])

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

  def epidemic_observables(self, parameters, epidemics):
    """See base class."""
    # Consider using the location of the peak as well.
    return {"epidemic_size": parameters[..., -1:]}
