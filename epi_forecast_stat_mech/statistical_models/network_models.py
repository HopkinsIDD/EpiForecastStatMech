# Lint as: python3
"""Statistical models based on Neural Networks."""
from typing import Any, Callable
import dataclasses

from epi_forecast_stat_mech.statistical_models import base
from epi_forecast_stat_mech.statistical_models import tree_util

from flax import nn
import jax
import jax.numpy as jnp

import tensorflow_probability as tfp
tfp = tfp.experimental.substrates.jax
tfd = tfp.distributions


class PerceptronModule(nn.Module):
  """Multi-layer perceptron network module."""

  def apply(
      self,
      inputs,
      output_size,
      hidden_layer_sizes=(64, 64),
      activation=jax.nn.relu,
      use_batch_norm=True,
  ):
    """Computes the output of a multi-layer perceptron give `inputs`."""
    x = inputs
    if use_batch_norm:
      x = nn.BatchNorm(x)
    for size in hidden_layer_sizes:
      x = nn.Dense(x, size)
      x = activation(x)
    return nn.Dense(x, output_size)


class LinearModule(
    PerceptronModule.partial(hidden_layer_sizes=(), activation=None)):
  pass


@dataclasses.dataclass
class NormalDistributionModel(base.StatisticalModel):
  """Statistical model based on normal distribution.

  Attributes:
    predict_module: flax.nn.Module that takes `inputs` and `output_size`
      arguments and returns array of shape `[batch, output_size]` that will be
      used to predict locations of the gaussian distributed `observations` and
      possibly scales, depending on whether `fixed_scale` is set.
    fixed_scale: fixed scale to use for modeling distribution over parameters.
      If `None`, then scale will be predicted by the `predict_module`.
    log_prior_fn: function that computes log_prior on a parameters of the
      `predict_module`.
    scale_eps: minimal value for the predicted scale if scale is predicted by
      `predict_module`.
  """
  predict_module: nn.Module = PerceptronModule
  log_prior_fn: Callable[..., Any] = None
  fixed_scale: float = None
  scale_eps: float = 1e-2

  def _output_size_and_unpack_fn(self, output_structure):
    output_array, unpack_fn = tree_util.pack(output_structure)
    output_size = output_array.shape[-1]
    if self.fixed_scale is None:
      output_size *= 2
    return output_size, unpack_fn

  def init_parameters(self, rng, inputs, output_structure):
    """Returns initial parameters generated at model construction time."""
    output_size, _ = self._output_size_and_unpack_fn(output_structure)
    return self.predict_module.init(rng, inputs, output_size)[1]

  def log_prior(self, parameters):
    """Returns the log probability of `parameters` based on priors.

    Args:
      parameters: parameters of the statistical model.

    Returns:
      log-probabilities for parameter.
    """
    if self.log_prior_fn is None:
      return 0.
    return self.log_prior_fn(parameters)

  def log_likelihood(self, parameters, covariates, observations):
    """Returns the log likelihood of `observations`."""
    posterior = self.predict(parameters, covariates, observations)
    res = jax.tree_multimap(lambda p, o: p.log_prob(o), posterior, observations)
    return res

  def predict(self, parameters, covariates, observations):
    """Predicts a distribution over `observations` based on `covariates`.

    Args:
      parameters: parameters of the statistical model.
      covariates: array representing covariates for each location.
      observations: structure of observations to be predicted.

    Returns:
      pytree of probability distributions over `observations` given the
      `parameters` of the statistical model and `covariates`.
    """
    output_size, unpack_fn = self._output_size_and_unpack_fn(observations)
    raw_predictions = self.predict_module.call(
        parameters, covariates, output_size)
    if self.fixed_scale is None:
      loc, raw_scale = jnp.split(raw_predictions, 2, -1)
      scale = jax.nn.softplus(raw_scale) + self.scale_eps
    else:
      loc = raw_predictions
      scale = self.fixed_scale * jnp.ones_like(loc)
    return jax.tree_multimap(
        lambda l, s: tfd.Normal(loc=l, scale=s),
        unpack_fn(loc), unpack_fn(scale))

  def linear_coefficients(self, parameters):
    dense_name = [x for x in parameters.keys() if "Dense" in x][0]
    kernel = parameters[dense_name]["kernel"]
    bias = parameters[dense_name]["bias"]
    if self.fixed_scale is None:
      kernel, _ = jnp.split(kernel, 2, -1)
      bias, _ = jnp.split(bias, 2, -1)
    return kernel, bias


