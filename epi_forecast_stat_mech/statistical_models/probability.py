# Lint as: python3
"""Common code for computing probabilities."""

import functools
import jax
import jax.numpy as jnp
from jax.scipy import special

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
  mask = observations >= 0
  observations = jnp.where(mask, observations, 0.)
  root_observations = jnp.sqrt(observations + .375)
  root_intensity = jnp.sqrt(intensity + .375)
  plugin_error_model = gaussian_with_softplus_scale_estimate(
      root_observations,
      axis=axis,
      min_scale=0.5,
      mean=root_intensity,
      softness=0.5)
  log_probs = plugin_error_model.log_prob(root_observations)
  return jnp.where(mask, log_probs, 0.)


# Normalizing factor so that `soft_laplace_log_prob` is a proper distribution.
LOG_SOFT_LAPLACE_INTEGRAL = 1.185495232349193


def soft_laplace_log_prob(x, scale=1):
  """Approximate Laplace log probability, differentiable near zero."""
  return (-jnp.sqrt((x / scale)**2 + 1) + 1  # 'soft' absolute value.
          - LOG_SOFT_LAPLACE_INTEGRAL
          - jnp.log(scale))

# Fun fact: sech(x)/pi is a prob. density.
# From this one could form alternatives to soft_laplace_log_prob.
# c.p. https://en.wikipedia.org/wiki/Hyperbolic_secant_distribution


def laplace_prior(parameters, scale_parameter=1.):
  return jax.tree_map(
      lambda x: tfd.Laplace(
          loc=jnp.zeros_like(x), scale=scale_parameter * jnp.ones_like(x)).
      log_prob(x), parameters)


@functools.partial(jnp.vectorize, signature='(k)->()')
def log_mixed_laplace(x, a=0., b=0.5, include_constants=False):
  """A scale-mixture of Laplace priors.

  Let S~InverseGamma(a, b); X_i |S=s ~[iid] Laplace(0, s).
  This computes the log density of the marginal on X.
  The choice a=0.5, b=0.5 is a reasonable proper prior, but the default,
  here is the choice a=0., b=0.5, and to omit the normalizing constants.
  """
  assert x.ndim == 1
  n = x.shape[-1]
  result = -(a + n) * jnp.log(b + jnp.sum(jnp.abs(x)))
  if include_constants:
    result += (
        a * jnp.log(b) - n * jnp.log(2.) - special.gammaln(a) +
        special.gammaln(a + n))
  return result


# c.p. log(cosh(x/delta))*delta.


def soft_abs(x, delta):
  return jnp.sqrt(delta**2 + x**2) - delta


@functools.partial(jnp.vectorize, signature='(k)->()')
def log_soft_mixed_laplace(x, a=0., b=0.5, delta=0.01, include_constants=False):
  """A softened scale-mixture of Laplace priors.

  Let S~InverseGamma(a, b); X_i |S=s ~[iid] Laplace(0, s).
  This computes the log density of the marginal on X, except that |x_i| terms
  are replaced with soft variants so that the function is twice continuously
  differentiable.
  """
  assert x.ndim == 1
  n = x.shape[-1]
  result = -(a + n) * jnp.log(b + jnp.sum(soft_abs(x, delta)))
  if include_constants:
    result += (
        a * jnp.log(b) - n * jnp.log(2.) - special.gammaln(a) +
        special.gammaln(a + n))
  return result


def apply_to_recursive_dict_by_last_key(last_key_mapper, d, default):
  return _rec_apply_to_recursive_dict_by_last_key(last_key_mapper, d, None,
                                                  default)


def _rec_apply_to_recursive_dict_by_last_key(last_key_mapper, d, last_key,
                                             default):
  if isinstance(d, dict):
    return {
        key: _rec_apply_to_recursive_dict_by_last_key(last_key_mapper, value,
                                                      key, default)
        for (key, value) in d.items()
    }
  fun = last_key_mapper.get(last_key, None)
  if fun is None:
    return default
  else:
    return fun(d)


def log_soft_mixed_laplace_on_kernels(flax_dict):
  return apply_to_recursive_dict_by_last_key(
      {'kernel': lambda x: log_soft_mixed_laplace(x.T)},
      flax_dict,
      0.)
