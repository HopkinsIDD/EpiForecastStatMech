# Lint as: python3
"""Tests for epi_forecast_stat_mech.statistical_models.network_models.py."""

from absl.testing import parameterized
from epi_forecast_stat_mech.statistical_models import network_models
import jax

absl.testing


class NetworkModelsTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(predict_module=network_models.PerceptronModule,
           dict_keys=None,
           num_locations=6,
           num_covariates=5,
           num_observations=3,
           fixed_scale=None),
      dict(predict_module=network_models.LinearModule,
           dict_keys=None,
           num_locations=6,
           num_covariates=5,
           num_observations=3,
           fixed_scale=None),
      dict(predict_module=network_models.PerceptronModule,
           dict_keys=('one', 'two'),
           num_locations=6,
           num_covariates=5,
           num_observations=3,
           fixed_scale=0.5),
  )
  def testShapes(
      self,
      predict_module,
      dict_keys,
      num_locations,
      num_covariates,
      num_observations,
      fixed_scale
  ):
    """Tests that `log_likelihood` and `predict` return expected shapes."""
    rng = jax.random.PRNGKey(42)
    c_rng, o_rng, init_rng = jax.random.split(rng, 3)
    covariates = jax.random.uniform(c_rng, (num_locations, num_covariates))
    observations = jax.random.uniform(o_rng, (num_locations, num_observations))
    if dict_keys is not None:
      observations = {key: observations for key in dict_keys}

    stat_model = network_models.NormalDistributionModel(
        predict_module=predict_module,
        fixed_scale=fixed_scale)
    params = stat_model.init_parameters(init_rng, covariates, observations)

    log_likelihood = stat_model.log_likelihood(params, covariates, observations)
    actual_shapes = jax.tree_map(lambda x: x.shape, log_likelihood)
    expected_shapes = jax.tree_map(lambda x: x.shape, observations)
    self.assertEqual(actual_shapes, expected_shapes)

    distributions_over_observations = stat_model.predict(
        params, covariates, observations)
    actual_shapes = jax.tree_map(
        lambda x: x.mean().shape, distributions_over_observations)
    expected_shapes = jax.tree_map(lambda x: x.shape, observations)
    self.assertEqual(actual_shapes, expected_shapes)


if __name__ == '__main__':
  absltest.main()
