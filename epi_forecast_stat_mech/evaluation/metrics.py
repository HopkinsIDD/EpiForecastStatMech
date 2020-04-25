# Lint as: python3
"""Metrics for evaluating models."""

import jax
import jax.numpy as jnp
import numpy as np


def quantile(trajectories, sample_axis=1, quantiles=np.linspace(0, 1, 11)):
  """Computes quantiles over sampled trajectories.

  Args:
    trajectories: an array of rolled out trajectories, typically of shape
      `[batch, samples, ...]`.
    sample_axis: the axis along corresponding to the samples for each element of
      the batch.
    quantiles: values in [0, 1], quantiles to compute.

  Returns:
    An array containing the quantiles. Given `trajectories` of shape
    `[batch, samples, ...]`, returns an array of shape
    `[len(quantiles), batch, ...]`.
  """
  # TODO(jamieas): consider transposing so that the quantile axis aligns with
  # the samples axis.
  return jnp.quantile(trajectories, quantiles, axis=sample_axis)


def target_quantile(trajectories, target, sample_axis=1):
  """Computes the quantile of `target` among `trajectories`.

  Args:
    trajectories: an array of rolled out trajectories, typically of shape
      `[batch, samples, ...]`.
    target: an array, typically of shape `[batch, ...]` i.e. the same shape as
      trajectories, but without dimension `sample_axis`.
    sample_axis: the axis along corresponding to the samples for each element of
      the batch.

  Returns:
    An array with the same shape as `target` containing the quantiles.
  """
  target_greater = jax.vmap(lambda y: target > y, sample_axis)(trajectories)
  return target_greater.mean(0)
