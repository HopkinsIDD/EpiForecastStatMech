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
    train_length=100,
    prediction_length=10,
):
  """Creates synthetic data."""
  np.random.seed(seed)  # TODO(shoyer): use np.random.RandomState
  beta_fn = functools.partial(sir_sim.generate_betas_many_cov2,
                              num_pred=num_important_cov,
                              num_not_pred=num_unimportant_cov)

  trajectories = sir_sim.generate_simulations(
      beta_fn,
      num_epidemics,
      num_time_steps=train_length+prediction_length)

  train_data, test_data = run_on_data.train_test_split_time(
      trajectories, trajectories.time[-prediction_length])

  return train_data, test_data


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

  train_data, test_data = run_on_data.train_test_split_time(
      data, data.canonical_split_time)

  return train_data, test_data


class TestHighLevelStatMech(absltest.TestCase):
  """Tests for StatMech high_level module."""

  def test_StatMechEstimator(self):
    """Verify we can fit and predict from StatMechEstimator."""
    num_samples = 11 # number of 'roll out' samples.

    train_data, test_data = create_synthetic_dataset()
    estimator = high_level.StatMechEstimator(train_steps=1000).fit(train_data)

    _ = estimator.mech_params.to_netcdf()
    predictions = estimator.predict(test_data, num_samples)
    self.assertCountEqual(['location', 'sample', 'time'], predictions.dims)
    np.testing.assert_array_equal(predictions.time, test_data.time)
    np.testing.assert_array_equal(train_data.location, predictions.location)
    self.assertLen(predictions.sample, num_samples)


class TestHighLevelRtLive(absltest.TestCase):
  """Tests for RtLive high_level module."""

  def test_RtLiveEstimator(self):
    """Verify we can fit and predict from RtLiveEstimator."""
    num_samples = 11 # number of 'roll out' samples.

    train_data, test_data = create_synthetic_dataset()
    estimator = high_level.RtLiveEstimator(gamma=1.0).fit(train_data)

    predictions = estimator.predict(test_data, num_samples)
    self.assertCountEqual(['location', 'sample', 'time'], predictions.dims)
    np.testing.assert_array_equal(predictions.time, test_data.time)
    np.testing.assert_array_equal(train_data.location, predictions.location)
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
    num_samples = 11  # number of 'roll out' samples.

    train_data, test_data = create_synthetic_dataset()
    estimator = high_level.get_estimator_dict()[estimator_name]
    estimator.fit(train_data)

    _ = estimator.mech_params.to_netcdf()
    _ = estimator.mech_params_hat.to_netcdf()
    predictions = estimator.predict(test_data, num_samples)
    self.assertCountEqual(['location', 'sample', 'time'], predictions.dims)
    np.testing.assert_array_equal(predictions.time, test_data.time)
    np.testing.assert_array_equal(train_data.location, predictions.location)
    self.assertLen(predictions.sample, num_samples)

  @parameterized.parameters(
      dict(estimator_name='LSML_Gaussian_PL_Linear_ObsEnc'),
      dict(estimator_name='LSML_VC_Linear_ObsEnc_plugin'),
      dict(estimator_name='LSML_Turner_Linear_ObsEnc_6wk'),
  )
  def test_EstimatorDictEstimatorWithCoef(self, estimator_name):
    """Verify we can fit and predict from the named estimator.

    This test requires mech_params as well as alpha and intercept.

    Args:
      estimator_name: a key into high_level.get_estimator_dict().
    """
    num_samples = 11  # number of 'roll out' samples.

    train_data, test_data = create_synthetic_dataset()
    estimator = high_level.get_estimator_dict()[estimator_name]
    estimator.fit(train_data)

    _ = estimator.alpha.to_netcdf()
    _ = estimator.intercept.to_netcdf()
    _ = estimator.mech_params.to_netcdf()
    predictions = estimator.predict(test_data, num_samples)
    self.assertCountEqual(['location', 'sample', 'time'], predictions.dims)
    np.testing.assert_array_equal(predictions.time, test_data.time)
    np.testing.assert_array_equal(train_data.location, predictions.location)
    self.assertLen(predictions.sample, num_samples)

  @parameterized.parameters(
      dict(
          estimator_name='iterative_randomforest__DynamicMultiplicative',
          run_it=False),
      dict(
          estimator_name='iterative_mean__DynamicBaselineSEIRModel',
          run_it=False),
  )
  def test_DynamicEstimatorDictEstimator(self, estimator_name, run_it):
    """Verify we can fit and predict from the named estimator.

    This test requires mech_params and mech_params_hat methods.

    Args:
      estimator_name: a key into high_level.get_estimator_dict().
    """
    num_samples = 11

    estimator = high_level.get_estimator_dict()[estimator_name]
    # I'm conditionally disabling this code to reduce timeout issues.
    if run_it:
      train_data, test_data = create_synthetic_dynamic_dataset()
      estimator.fit(train_data)

      _ = estimator.mech_params.to_netcdf()
      _ = estimator.mech_params_hat.to_netcdf()
      predictions = estimator.predict(test_data, num_samples)
      self.assertCountEqual(['location', 'sample', 'time'], predictions.dims)
      np.testing.assert_array_equal(predictions.time, test_data.time)
      np.testing.assert_array_equal(train_data.location, predictions.location)
      self.assertLen(predictions.sample, num_samples)


if __name__ == '__main__':
  absltest.main()
