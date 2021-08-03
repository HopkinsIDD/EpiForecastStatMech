# Lint as: python3
"""Tests for epi_forecast_stat_mech.evaluation.metrics."""

import functools
import itertools

from absl.testing import parameterized
from absl.testing import absltest

from epi_forecast_stat_mech.evaluation import metrics

import jax
from jax.config import config
import numpy as np

import tensorflow_probability

tfp = tensorflow_probability.substrates.jax
tfd = tfp.distributions

config.parse_flags_with_absl()  # Necessary for running on TPU.


class MetricsTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(
          metric=functools.partial(
              metrics.quantile, quantiles=np.linspace(0, 1, 11)),
          batch_size=23,
          nsamples=37,
          rollout_shape=(2, 3, 4),
          expected_shape=(11, 23) + (2, 3, 4),
          seed=0),
      dict(
          metric=functools.partial(
              metrics.target_quantile, target=np.zeros([11, 101])),
          batch_size=11,
          nsamples=400,
          rollout_shape=(101,),
          expected_shape=(11, 101),
          seed=1),
  )
  def testMetricsShape(
      self, metric, batch_size, nsamples, rollout_shape, expected_shape, seed):
    trajectories = jax.random.normal(jax.random.PRNGKey(seed),
                                     (batch_size, nsamples) + rollout_shape)
    metric_value = metric(trajectories)
    self.assertEqual(expected_shape, metric_value.shape)

  @parameterized.parameters(
      dict(batch_size=23,
           nsamples=1000,
           rollout_shape=(11,),
           quantiles=np.linspace(0, 1, 11),
           seed=0),
      dict(batch_size=11,
           nsamples=10000,
           rollout_shape=(3,),
           quantiles=np.linspace(0, 1, 7),
           seed=1)
  )
  def testQuantileAndTargetQuantile(
      self, batch_size, nsamples, rollout_shape, quantiles, seed):
    trajectories = jax.random.normal(jax.random.PRNGKey(seed),
                                     (batch_size, nsamples) + rollout_shape)
    targets = metrics.quantile(trajectories, quantiles=quantiles)
    for target, quantile in zip(targets, quantiles):
      np.testing.assert_allclose(quantile,
                                 metrics.target_quantile(trajectories, target),
                                 atol=.01)

if __name__ == '__main__':
  absltest.main()
