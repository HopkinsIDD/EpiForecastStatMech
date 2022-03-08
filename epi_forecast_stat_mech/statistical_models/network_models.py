# Lint as: python3
"""Statistical models based on Neural Networks."""
import dataclasses
from typing import Any, Callable
from epi_forecast_stat_mech.statistical_models import base
from epi_forecast_stat_mech.statistical_models import probability as stat_prob
from epi_forecast_stat_mech.statistical_models import tree_util

from flax.deprecated import nn
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
      batch_norm_style='once'
  ):
    """Computes the output of a multi-layer perceptron give `inputs`."""
    x = inputs
    if batch_norm_style == 'none' or batch_norm_style == 'once':
      if batch_norm_style == 'once':
        x = nn.BatchNorm(x)
      for size in hidden_layer_sizes:
        x = nn.Dense(x, size)
        x = activation(x)
      return nn.Dense(x, output_size)
    elif batch_norm_style == 'layerwise_whiten':
      for size in hidden_layer_sizes:
        x = nn.BatchNorm(x, bias=False, scale=False)
        x = nn.Dense(x, size)
        x = activation(x)
      x = nn.BatchNorm(x, bias=False, scale=False)
      return nn.Dense(x, output_size)
    else:
      raise ValueError(f'Unexpected batch_norm_style: {batch_norm_style}')


class PlainLinearModule(
    PerceptronModule.partial(
        hidden_layer_sizes=(), activation=None, batch_norm_style='none')):
  pass


class LinearModule(
    PerceptronModule.partial(
        hidden_layer_sizes=(), activation=None, batch_norm_style='once')):
  pass


@dataclasses.dataclass
class NormalDistributionModel(base.StatisticalModel):
  """Statistical model based on normal distribution.

  Attributes:
    predict_module: flax.deprecated.nn.Module that takes `inputs` and `output_size`
      arguments and returns array of shape `[batch, output_size]` that will be
      used to predict locations of the gaussian distributed `observations` and
      possibly scales, depending on whether `error_model` is 'full'.
    error_model: (string) Either 'full' to represent using the predict_module
      to estimate the error-scale (heteroscedastic). Or 'plugin' to
      represent that the scale should be based on a homoscedastic plugin.
    log_prior_fn: function that computes log_prior on a parameters of the
      `predict_module`.
    scale_eps: minimal value for the predicted scale if scale.
  """
  predict_module: nn.Module = PerceptronModule
  log_prior_fn: Callable[..., Any] = None
  error_model: str = 'full'
  scale_eps: float = 1e-2

  def _output_size_and_unpack_fn(self, output_structure):
    output_array, unpack_fn = tree_util.pack(output_structure)
    output_size = output_array.shape[-1]
    if self.error_model == 'full':
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
    """Returns the log likelihood of `observations`.

    Args:
      parameters: parameters of the statistical model.
      covariates: A numpy array of shape "location" x "static_covariate".
      observations: A tree of mech_params that we want to explain.

    Returns:
      A log-likelihood. If self.error_model == 'full', the return is a tree of
      the shape of observations (the log_likelihood of each observation). In the
      'plugin' case, the return is a scalar (the sum).
    """
    if self.error_model == 'full':
      posterior = self.predict(parameters, covariates, observations)
      res = tree_util.tree_multimap(
          lambda p, o: p.log_prob(o), posterior, observations)
      return res
    elif self.error_model == 'plugin':
      loc, _ = self.get_loc_scale(parameters, covariates, observations)
      tree_error = tree_util.tree_multimap(
          lambda hat, o: o - hat, loc, observations)
      error, _ = tree_util.pack(tree_error)
      logprob = stat_prob.gaussian_error_logprob_with_bottom_scale(
          error, self.scale_eps, axis=0)
      return logprob
    else:
      raise ValueError(f'unexpected error_model: {self.error_model}')

  def get_loc_scale(self, parameters, covariates, observations):
    """Computes loc and scale for `observations` based on `covariates`.

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
    if self.error_model == 'full':
      loc, raw_scale = jnp.split(raw_predictions, 2, -1)
      scale = jax.nn.softplus(raw_scale) + self.scale_eps
      return unpack_fn(loc), unpack_fn(scale)
    elif self.error_model == 'plugin':
      loc = raw_predictions
      return unpack_fn(loc), None
    else:
      raise ValueError(f'unexpected error_model: {self.error_model}')

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
    loc, scale = self.get_loc_scale(parameters, covariates, observations)
    return tree_util.tree_multimap(
        lambda l, s: tfd.Normal(loc=l, scale=s),
        loc, scale)

  def linear_coefficients(self, parameters):
    dense_name = [x for x in parameters.keys() if 'Dense' in x][0]
    kernel = parameters[dense_name]['kernel']
    bias = parameters[dense_name]['bias']
    if self.error_model == 'full':
      kernel, _ = jnp.split(kernel, 2, -1)
      bias, _ = jnp.split(bias, 2, -1)
    return kernel, bias
