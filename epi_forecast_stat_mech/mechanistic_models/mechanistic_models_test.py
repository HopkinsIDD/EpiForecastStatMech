# Lint as: python3
"""Tests for epi_forecast_stat_mech.mechanistic_models.mechanistic_models."""

import functools
from absl.testing import absltest
from absl.testing import parameterized

import jax
from jax.config import config
import numpy as np

from epi_forecast_stat_mech import sir_sim
from epi_forecast_stat_mech import utils
from epi_forecast_stat_mech.evaluation import run_on_data
from epi_forecast_stat_mech.mechanistic_models import mechanistic_models

config.parse_flags_with_absl()  # Necessary for running on TPU.

EpidemicsRecord = mechanistic_models.EpidemicsRecord


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
      dict(
          mech_model_cls=mechanistic_models.GaussianModel,
          observed_duration=10,
          trajectory_length=20),
      dict(
          mech_model_cls=mechanistic_models.ViboudChowellModel,
          observed_duration=13,
          trajectory_length=24),
      dict(
          mech_model_cls=mechanistic_models.GaussianModelPseudoLikelihood,
          observed_duration=10,
          trajectory_length=20),
      dict(
          mech_model_cls=mechanistic_models.ViboudChowellModelPseudoLikelihood,
          observed_duration=13,
          trajectory_length=24),
      dict(
          mech_model_cls=mechanistic_models.StepBasedViboudChowellModel,
          observed_duration=13,
          trajectory_length=24),
      dict(
          mech_model_cls=mechanistic_models.StepBasedGaussianModel,
          observed_duration=13,
          trajectory_length=24),
      dict(
          mech_model_cls=mechanistic_models.StepBasedMultiplicativeGrowthModel,
          observed_duration=13,
          trajectory_length=24),
  )
  def testShapes(self, mech_model_cls, observed_duration, trajectory_length):
    """Tests that mechanistic models methods return values of expected shape."""
    mech_model = mech_model_cls()
    mech_model_params = mech_model.init_parameters()
    rng = jax.random.PRNGKey(42)
    observed_epidemics = EpidemicsRecord(
        np.arange(observed_duration).astype(np.float32),
        np.arange(observed_duration).astype(np.float32),
        np.cumsum(np.arange(observed_duration).astype(np.float32)),
        np.zeros((observed_duration, 1)))
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

  @parameterized.parameters(
      dict(mech_model_cls=mechanistic_models.StepBasedViboudChowellModel,
           observed_duration=13, trajectory_length=24),
      dict(mech_model_cls=mechanistic_models.StepBasedGaussianModel,
           observed_duration=13, trajectory_length=24),
      dict(mech_model_cls=mechanistic_models.StepBasedMultiplicativeGrowthModel,
           observed_duration=13, trajectory_length=24),
  )
  def testTimeDependentModels(
      self,
      mech_model_cls,
      observed_duration,
      trajectory_length
  ):
    """Test output shapes for models that allow time dependent parameters."""
    mech_model = mech_model_cls()
    mech_model_params = np.tile(
        np.expand_dims(mech_model.init_parameters(), 0),
        (observed_duration + trajectory_length, 1))
    rng = jax.random.PRNGKey(42)
    observed_epidemics = EpidemicsRecord(
        np.arange(observed_duration).astype(np.float32),
        np.arange(observed_duration).astype(np.float32),
        np.cumsum(np.arange(observed_duration).astype(np.float32)),
        np.zeros((observed_duration, 0)))
    predicted_epidemics_trajectory = mech_model.predict(
        mech_model_params, rng, observed_epidemics, trajectory_length)
    actual_shape = predicted_epidemics_trajectory.shape
    expected_shape = (observed_duration + trajectory_length,)
    self.assertEqual(actual_shape, expected_shape)

  @parameterized.parameters(
      dict(mech_model_cls=mechanistic_models.DynamicMultiplicativeGrowthModel),
      # A second copy makes this work [until there are two].
      dict(mech_model_cls=mechanistic_models.DynamicMultiplicativeGrowthModel),
  )
  def testDynamic(self, mech_model_cls):
    """Tests that dynamic mechanistic models function."""
    np.random.seed(2020052702)
    beta_fn = functools.partial(
        sir_sim.generate_betas_many_cov2, num_pred=3, num_not_pred=7)
    data = sir_sim.generate_social_distancing_simulations(
        beta_fn, sir_sim.gen_social_distancing_weight, 25, 200)
    data = data.sel(
        time=((data.new_infections.sum('location') >= 1).cumsum('time') >= 1))
    data = data.sel(location=(data.new_infections.sum('time') >= 100))
    train, _ = run_on_data.train_test_split_time(data,
                                                 data.canonical_split_time)

    rng1, rng = jax.random.split(jax.random.PRNGKey(42))
    mech_model = mech_model_cls(rng1, data.dynamic_covariate)
    mech_model_params = mech_model.init_parameters()
    all_observed_epidemics = mechanistic_models.pack_epidemics_record_tuple(
        train)
    observed_epidemic, unused_record_rest = utils.split_along_axis(
        all_observed_epidemics, 0, 1)
    dynamic_covariates0 = data.dynamic_covariates.isel(location=0).transpose(
        'time', 'dynamic_covariate').data

    model_log_prob = mech_model.log_likelihood(mech_model_params,
                                               observed_epidemic)
    actual_shape = model_log_prob.shape
    expected_shape = (train.sizes['time'],)
    self.assertEqual(actual_shape, expected_shape)

    predicted_epidemics_trajectory = mech_model.predict(mech_model_params, rng,
                                                        observed_epidemic,
                                                        dynamic_covariates0)
    actual_shape = predicted_epidemics_trajectory.shape
    expected_shape = (data.sizes['time'],)
    self.assertEqual(actual_shape, expected_shape)


if __name__ == '__main__':
  absltest.main()
