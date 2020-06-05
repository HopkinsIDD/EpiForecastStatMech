# Lint as: python3
"""Tests for epi_forecast_stat_mech.high_level."""

import functools

from absl.testing import absltest

from epi_forecast_stat_mech import sparse
from epi_forecast_stat_mech import sparse_estimator
from epi_forecast_stat_mech.tests import test_high_level

import numpy as np


class TestHighLevelSparseEstimator(absltest.TestCase):
  """Tests for Sparse high_level module."""

  def test_SparseEstimator(self):
    """Verify we can fit and predict from SparseEstimator."""
    prediction_length = 10
    num_samples = 11

    data = test_high_level.create_synthetic_dataset(
        num_epidemics=50, num_time_steps=100)
    # These arguments are chosen to make the test faster.
    estimator = sparse_estimator.SparseEstimator(
        initializer=sparse.predefined_constant_initializer,
        optimizer=functools.partial(sparse._adam_optim, max_iter=10),
        penalty_factor_grid=np.exp(
            np.linspace(np.log(.1), np.log(1000.), num=5))).fit(data)
    predictions = estimator.predict(prediction_length, num_samples)
    self.assertCountEqual(['location', 'sample', 'time'], predictions.dims)
    self.assertLen(predictions.time, prediction_length)
    np.testing.assert_array_equal(data.location, predictions.location)
    self.assertLen(predictions.sample, num_samples)
    _ = estimator.alpha.to_netcdf()
    _ = estimator.intercept.to_netcdf()
    _ = estimator.mech_params.to_netcdf()
    _ = estimator.mech_params_hat.to_netcdf()


if __name__ == '__main__':
  absltest.main()
