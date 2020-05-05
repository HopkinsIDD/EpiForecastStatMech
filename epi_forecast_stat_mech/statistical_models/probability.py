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

# Normalizing factor so that `soft_laplace_log_prob` is a proper distribution.
LOG_SOFT_LAPLACE_INTEGRAL = 1.185495232349193


def soft_laplace_log_prob(x, scale=1):
  """Approximate Laplace log probability, differentiable near zero."""
  return (-jnp.sqrt((x / scale)**2 + 1) + 1  # 'soft' absolute value.
          - LOG_SOFT_LAPLACE_INTEGRAL
          - jnp.log(scale))
