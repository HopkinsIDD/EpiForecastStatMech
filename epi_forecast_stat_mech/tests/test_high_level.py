# Lint as: python3
"""Tests for epi_forecast_stat_mech.high_level."""

import collections
import functools

from absl.testing import absltest

from epi_forecast_stat_mech import high_level
from epi_forecast_stat_mech import sir_sim

from jax.config import config
import numpy as np
import sklearn

config.parse_flags_with_absl()  # Necessary for running on TPU.


def create_synthetic_dataset(
    seed=0,
    num_epidemics=50,
    num_important_cov=1,
    num_unimportant_cov=2,
    num_time_steps=100,
):
  """Creates synthetic data."""
  np.random.seed(seed)  # TODO(shoyer): use np.random.RandomState
  beta_fn = functools.partial(sir_sim.generate_betas_many_cov2,
                              num_pred=num_important_cov,
                              num_not_pred=num_unimportant_cov)
  num_simulations = 1
  trajectories = sir_sim.generate_simulations(
      beta_fn,
      num_simulations, num_epidemics,
      num_time_steps=num_time_steps)
  trajectories = trajectories.squeeze('sample')
  return trajectories


class TestHighLevelStatMech(absltest.TestCase):
  """Tests for StatMech high_level module."""

  def test_StatMechEstimator(self):
    """Verify we can fit and predict from StatMechEstimator."""
    prediction_length = 10
    num_samples = 11

    data = create_synthetic_dataset(num_epidemics=50, num_time_steps=100)
    estimator = high_level.StatMechEstimator(train_steps=1000).fit(data)

    _ = estimator.mech_params.to_netcdf()
    predictions = estimator.predict(prediction_length, num_samples)
    self.assertCountEqual(['location', 'sample', 'time'], predictions.dims)
    self.assertLen(predictions.time, prediction_length)
    np.testing.assert_array_equal(data.location, predictions.location)
    self.assertLen(predictions.sample, num_samples)


class TestHighLevelRtLive(absltest.TestCase):
  """Tests for RtLive high_level module."""

  def test_RtLiveEstimator(self):
    """Verify we can fit and predict from RtLiveEstimator."""
    prediction_length = 10
    num_samples = 11

    data = create_synthetic_dataset(num_epidemics=50, num_time_steps=100)
    estimator = high_level.RtLiveEstimator(gamma=1.0).fit(data)

    predictions = estimator.predict(prediction_length, num_samples)
    self.assertCountEqual(['location', 'sample', 'time'], predictions.dims)
    self.assertLen(predictions.time, prediction_length)
    np.testing.assert_array_equal(data.location, predictions.location)
    self.assertLen(predictions.sample, num_samples)


class TestHighLevelIterativeEstimator(absltest.TestCase):
  """Tests for Iterative high_level module."""

  def test_IterativeEstimator(self):
    """Verify we can fit and predict from IterativeEstimator."""
    prediction_length = 10
    num_samples = 11

    data = create_synthetic_dataset(num_epidemics=50, num_time_steps=100)
    mean_estimators = collections.defaultdict(
        lambda: sklearn.dummy.DummyRegressor(strategy='mean'))
    estimator = high_level.IterativeEstimator(
        stat_estimators=mean_estimators).fit(data)

    _ = estimator.mech_params.to_netcdf()
    _ = estimator.mech_params_hat.to_netcdf()
    predictions = estimator.predict(prediction_length, num_samples)
    self.assertCountEqual(['location', 'sample', 'time'], predictions.dims)
    self.assertLen(predictions.time, prediction_length)
    np.testing.assert_array_equal(data.location, predictions.location)
    self.assertLen(predictions.sample, num_samples)


class TestGetEstimatorDict(absltest.TestCase):
  """Tests for get_estimator_dict."""

  def test_get_estimator_dict(self):
    _ = high_level.get_estimator_dict()


if __name__ == '__main__':
  absltest.main()
