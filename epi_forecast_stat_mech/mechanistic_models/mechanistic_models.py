# Lint as: python3
"""Mechanistic models for forecasting epidemics."""

import abc
from typing import Callable, Union
import dataclasses


import jax
import jax.numpy as jnp
import numpy as np

import tensorflow_probability as tfp
tfp = tfp.experimental.substrates.jax
tfd = tfp.distributions

Array = Union[float, jnp.DeviceArray, np.ndarray]


@dataclasses.dataclass
class FastPoisson:
  """A Poisson distribution that uses a Normal approximation for large rate."""

  rate: Array
  rate_split: float = 100.

  def sample(self, sample_shape, rng, intensity_split=100.0):
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

  @staticmethod
  def init_parameters():
    """Returns reasonable `parameters` for an initial guess."""
    return jnp.log(jnp.asarray([2., .9, .9, 250000.]))

  def log_likelihood(self, parameters, epidemics):
    """Returns the log likelihood of `epidemics` given `parameters`."""
    def _log_prob_step(cumulative_cases, new_cases):
      intensity = self.intensity(parameters, cumulative_cases)
      log_prob = jnp.squeeze(
          self.new_infection_distribution(intensity).log_prob(new_cases))
      return cumulative_cases + new_cases, log_prob

    start_cases = epidemics.cumulative_infections[0]
    trajectory = epidemics.infections_over_time[1:]
    _, log_probs = jax.lax.scan(_log_prob_step, start_cases, trajectory)
    # we include 0 to indicate that we treat the first step as given.
    # Note: this might make log-likelihood unfair compared to other models that
    # are evaluated on the first step as well.
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
    cumulative_cases = jnp.sum(observed_epidemics.infections_over_time)
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
    return {'epidemic_size': parameters[..., -1:]}


@dataclasses.dataclass
class GaussianModel(MechanisticModel):
  """Mechanistic model based on gaussian shape of the epidemics."""

  # A callable that takes a single argument `mean` and returns an object
  # that has a `.predict` and `log_prob` method. Typically, this is a TensorFlow
  # Probability distribution.
  new_infection_distribution: Callable = FastPoisson  #pylint: disable=g-bare-generic

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

  @staticmethod
  def init_parameters():
    """Returns reasonable `parameters` for an initial guess."""
    return jnp.asarray([100., np.log(100.), np.log(1000.)])

  def log_likelihood(self, parameters, epidemics):
    """Returns the log likelihood of `epidemics` given `parameters`."""
    def _log_prob_step(t, new_cases):
      intensity = self.intensity(parameters, t)
      log_prob = jnp.squeeze(
          self.new_infection_distribution(intensity).log_prob(new_cases))
      return t + 1, log_prob

    init_time = epidemics.t[0]
    trajectory = epidemics.infections_over_time
    _, log_probs = jax.lax.scan(_log_prob_step, init_time, trajectory)
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
    return {'epidemic_size': parameters[..., -1:]}
