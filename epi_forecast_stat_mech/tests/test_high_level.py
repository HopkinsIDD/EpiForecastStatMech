# Lint as: python3
"""Tests for epi_forecast_stat_mech.high_level."""

import collections
import functools

from absl.testing import absltest
from absl.testing import parameterized

from epi_forecast_stat_mech import high_level
from epi_forecast_stat_mech import sir_sim
from epi_forecast_stat_mech.evaluation import run_on_data

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

  trajectories = sir_sim.generate_simulations(
      beta_fn,
      num_epidemics,
      num_time_steps=num_time_steps)

  return trajectories


def create_synthetic_dynamic_dataset(
    seed=0,
    num_epidemics=25,
    num_important_cov=1,
    num_unimportant_cov=2,
    num_time_steps=200,
):
  """Creates synthetic dynamic data."""
  np.random.seed(seed)
  beta_fn = functools.partial(
      sir_sim.generate_betas_many_cov2,
      num_pred=num_important_cov,
      num_not_pred=num_unimportant_cov)
  data = sir_sim.generate_social_distancing_simulations(
      beta_fn, sir_sim.gen_social_distancing_weight, num_epidemics,
      num_time_steps)
  data = data.sel(
      time=((data.new_infections.sum('location') >= 1).cumsum('time') >= 1))
  data = data.sel(location=(data.new_infections.sum('time') >= 100))
  return data


class TestHighLevelStatMech(absltest.TestCase):
  """Tests for StatMech high_level module."""

  def test_StatMechEstimator(self):
    """Verify we can fit and predict from StatMechEstimator."""
    prediction_length = 10
    num_samples = 11 # number of 'roll out' samples.

    data = create_synthetic_dataset(num_time_steps=100)
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
    num_samples = 11 # number of 'roll out' samples.

    data = create_synthetic_dataset(num_time_steps=100)
    estimator = high_level.RtLiveEstimator(gamma=1.0).fit(data)

    predictions = estimator.predict(prediction_length, num_samples)
    self.assertCountEqual(['location', 'sample', 'time'], predictions.dims)
    self.assertLen(predictions.time, prediction_length)
    np.testing.assert_array_equal(data.location, predictions.location)
    self.assertLen(predictions.sample, num_samples)


class TestGetEstimatorDict(absltest.TestCase):
  """Tests for get_estimator_dict."""

  def test_get_estimator_dict(self):
    _ = high_level.get_estimator_dict()


class TestEstimatorDictEstimator(parameterized.TestCase):
  """Tests for high_level.get_estimator_dict estimators."""

  @parameterized.parameters(
      dict(estimator_name='iterative_randomforest__VC'),
      dict(estimator_name='iterative_mean__Gaussian_PL'),
  )
  def test_EstimatorDictEstimator(self, estimator_name):
    """Verify we can fit and predict from the named estimator.

    This test requires mech_params and mech_params_hat methods.

    Args:
      estimator_name: a key into high_level.get_estimator_dict().
    """
    prediction_length = 10
    num_samples = 11  # number of 'roll out' samples.

    data = create_synthetic_dataset(num_time_steps=100)
    estimator = high_level.get_estimator_dict()[estimator_name]
    estimator.fit(data)

    _ = estimator.mech_params.to_netcdf()
    _ = estimator.mech_params_hat.to_netcdf()
    predictions = estimator.predict(prediction_length, num_samples)
    self.assertCountEqual(['location', 'sample', 'time'], predictions.dims)
    self.assertLen(predictions.time, prediction_length)
    np.testing.assert_array_equal(data.location, predictions.location)
    self.assertLen(predictions.sample, num_samples)

  @parameterized.parameters(
      dict(estimator_name='None_VC_Linear'),
      dict(estimator_name='Laplace_Gaussian_PL_Linear'),
      dict(estimator_name='None_MultiplicativeGrowth_Linear'),
  )
  def test_EstimatorDictEstimatorWithCoef(self, estimator_name):
    """Verify we can fit and predict from the named estimator.

    This test requires mech_params as well as alpha and intercept.

    Args:
      estimator_name: a key into high_level.get_estimator_dict().
    """
    prediction_length = 10
    num_samples = 11  # number of 'roll out' samples.

    data = create_synthetic_dataset(num_time_steps=100)
    estimator = high_level.get_estimator_dict()[estimator_name]
    estimator.fit(data)

    _ = estimator.alpha.to_netcdf()
    _ = estimator.intercept.to_netcdf()
    _ = estimator.mech_params.to_netcdf()
    predictions = estimator.predict(prediction_length, num_samples)
    self.assertCountEqual(['location', 'sample', 'time'], predictions.dims)
    self.assertLen(predictions.time, prediction_length)
    np.testing.assert_array_equal(data.location, predictions.location)
    self.assertLen(predictions.sample, num_samples)

  @parameterized.parameters(
      dict(estimator_name='iterative_mean__DynamicMultiplicative'),
      dict(estimator_name='iterative_randomforest__DynamicMultiplicative'),
  )
  def test_DynamicEstimatorDictEstimator(self, estimator_name):
    """Verify we can fit and predict from the named estimator.

    This test requires mech_params and mech_params_hat methods.

    Args:
      estimator_name: a key into high_level.get_estimator_dict().
    """
    num_samples = 11

    data = create_synthetic_dynamic_dataset()
    train, _ = run_on_data.train_test_split_time(data,
                                                 data.canonical_split_time)
    estimator = high_level.get_dynamic_estimator_dict()[estimator_name]
    estimator.fit(train)

    _ = estimator.mech_params.to_netcdf()
    _ = estimator.mech_params_hat.to_netcdf()
    predictions = estimator.predict(
        data.dynamic_covariates, num_samples, include_observed=False)
    self.assertCountEqual(['location', 'sample', 'time'], predictions.dims)
    self.assertLen(predictions.time, data.sizes['time'] - train.sizes['time'])
    np.testing.assert_array_equal(data.location, predictions.location)
    self.assertLen(predictions.sample, num_samples)


if __name__ == '__main__':
  absltest.main()
