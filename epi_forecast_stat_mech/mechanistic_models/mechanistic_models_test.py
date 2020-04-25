# Lint as: python3
"""Tests for epi_forecast_stat_mech.mechanistic_models.mechanistic_models."""

import collections
from absl.testing import parameterized
from epi_forecast_stat_mech.mechanistic_models import mechanistic_models
import jax
import numpy as np

from absl.testing import absltest


EpidemicsRecord = collections.namedtuple(
    'EpidemicsRecord',
    ['t', 'infections_over_time', 'cumulative_infections'])


class PoissonSampleTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(rate=100.0, intensity_split=50.0),
      dict(rate=100.0, intensity_split=500.0),
  )
  def test(self, rate, intensity_split):
    rng = jax.random.PRNGKey(0)
    sample_shape = 1000
    distribution = mechanistic_models.FastPoisson(rate, intensity_split)
    samples = distribution.sample(sample_shape, rng)
    # some basic sanity tests
    self.assertBetween(samples.mean(), 90, 110)
    self.assertBetween(samples.var(), 90, 110)
    self.assertTrue((samples >= 0).all())
    self.assertTrue((samples < 1e6).all())


class MechanisticModelsTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(mech_model_cls=mechanistic_models.GaussianModel,
           observed_duration=10, trajectory_length=20),
      dict(mech_model_cls=mechanistic_models.ViboudChowellModel,
           observed_duration=13, trajectory_length=24),
      )
  def testShapes(self, mech_model_cls, observed_duration, trajectory_length):
    """Tests that mechanistic models methods return values of expected shape."""
    mech_model = mech_model_cls()
    mech_model_params = mech_model_cls.init_parameters()
    rng = jax.random.PRNGKey(42)
    observed_epidemics = EpidemicsRecord(
        np.arange(observed_duration).astype(np.float32),
        np.arange(observed_duration).astype(np.float32),
        np.cumsum(np.arange(observed_duration).astype(np.float32)))
    predicted_epidemics_trajectory = mech_model.predict(
        mech_model_params, rng, observed_epidemics, trajectory_length)
    actual_shape = predicted_epidemics_trajectory.shape
    expected_shape = (observed_duration + trajectory_length,)
    self.assertEqual(actual_shape, expected_shape)

    model_log_prob = mech_model.log_likelihood(
        mech_model_params, observed_epidemics)
    actual_shape = model_log_prob.shape
    expected_shape = (observed_duration,)
    self.assertEqual(actual_shape, expected_shape)


if __name__ == '__main__':
  absltest.main()
