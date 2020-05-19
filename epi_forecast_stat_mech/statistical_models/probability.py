# Lint as: python3
"""Common code for computing probabilities."""

import jax
import jax.numpy as jnp

import tensorflow_probability
tfp = tensorflow_probability.experimental.substrates.jax
tfd = tfp.distributions


def softplus(x, floor, softness=1):
  """See https://en.wikipedia.org/wiki/Rectifier_(neural_networks)#Softplus."""
  return jax.nn.softplus((x - floor) / softness) * softness + floor


def softplus_scale_estimate(
    x, axis, min_scale, softness=1):
  """Returns the standard deviation of `x` after enforcing a minimum."""
  scale = x.std(axis=axis, keepdims=True)
  return softplus(scale, min_scale, softness)


def gaussian_with_softplus_scale_estimate(
    x, axis, min_scale, mean=0, softness=1):
  """Returns a Gaussian distribution whose variance is estimated using `x`."""
  scale_estimate = softplus_scale_estimate(x - mean, axis, min_scale, softness)
  return tfd.Normal(mean, jax.lax.stop_gradient(scale_estimate))


def pseudo_poisson_log_probs(intensity, observations, axis=0):
  """Compute pseudo log-probs for observations as overdispersed Poisson's.

  The overdispersion scale is set along axis for each such "column" in a plugin
  manner (hence the pseudo; also the result is a density for an imaginary
  Poisson-like continuous density, not a discrete distribution).

  See https://colab.sandbox.google.com/drive/1hzI3rbQ-mtrRgG1lSuk3YWy0WcoDVYI-
  for references to Anscombe, etc., to explain the sqrt.

  Args:
    intensity: An array of (pseudo)-Poisson intensities.
    observations: A (matched) array of counts.
    axis: (default 0) The axis along-which the overdispersion scale will be
      estimated and plugged in.
  Returns:
    log_probs: An entry for each entry of observations.
  """
  root_observations = jnp.sqrt(observations + .375)
  root_intensity = jnp.sqrt(intensity + .375)
  plugin_error_model = gaussian_with_softplus_scale_estimate(
      root_observations,
      axis=axis,
      min_scale=0.5,
      mean=root_intensity,
      softness=0.5)
  log_probs = plugin_error_model.log_prob(root_observations)
  return log_probs


# Normalizing factor so that `soft_laplace_log_prob` is a proper distribution.
LOG_SOFT_LAPLACE_INTEGRAL = 1.185495232349193


def soft_laplace_log_prob(x, scale=1):
  """Approximate Laplace log probability, differentiable near zero."""
  return (-jnp.sqrt((x / scale)**2 + 1) + 1  # 'soft' absolute value.
          - LOG_SOFT_LAPLACE_INTEGRAL
          - jnp.log(scale))
