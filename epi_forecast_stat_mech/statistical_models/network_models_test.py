# Lint as: python3
"""Tests for epi_forecast_stat_mech.statistical_models.network_models.py."""

from absl.testing import absltest
from absl.testing import parameterized
from epi_forecast_stat_mech.statistical_models import network_models
from epi_forecast_stat_mech.statistical_models import tree_util
import jax
import jax.numpy as jnp
from jax.config import config

config.parse_flags_with_absl()  # Necessary for running on TPU.


class NetworkModelsTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(predict_module=network_models.PerceptronModule,
           dict_keys=None,
           num_locations=7,
           num_covariates=5,
           num_observations=3,
           error_model='full'),
      dict(predict_module=network_models.LinearModule,
           dict_keys=None,
           num_locations=7,
           num_covariates=5,
           num_observations=3,
           error_model='full'),
      dict(predict_module=network_models.PerceptronModule,
           dict_keys=('one', 'two'),
           num_locations=7,
           num_covariates=5,
           num_observations=3,
           error_model='plugin'),
  )
  def testShapes(
      self,
      predict_module,
      dict_keys,
      num_locations,
      num_covariates,
      num_observations,
      error_model
  ):
    """Tests that `log_likelihood` and `predict` return expected shapes."""
    rng = jax.random.PRNGKey(42)
    c_rng, o_rng, init_rng = jax.random.split(rng, 3)
    covariates = jax.random.uniform(c_rng, (num_locations, num_covariates))
    observations = jax.random.uniform(o_rng, (num_locations, num_observations))
    if dict_keys is not None:
      observations = {key: observations for key in dict_keys}

    if error_model == 'full':
      scale_eps = 1E-2
    else:
      # The length of this needs to agree with the flattened number of
      # "observations" per row. I.e. flattened observations is (7, 6)
      # for the 'plugin' test because the two dict keys double the
      # three observations per location.
      scale_eps = jnp.asarray([.1, .2, .3, .4, .5, .6])
    stat_model = network_models.NormalDistributionModel(
        predict_module=predict_module,
        error_model=error_model,
        scale_eps=scale_eps)
    params = stat_model.init_parameters(init_rng, covariates, observations)

    log_likelihood = stat_model.log_likelihood(params, covariates, observations)

    if error_model == 'full':
      actual_shapes = tree_util.tree_map(lambda x: x.shape, log_likelihood)
      expected_shapes = tree_util.tree_map(lambda x: x.shape, observations)
      self.assertEqual(actual_shapes, expected_shapes)
      distributions_over_observations = stat_model.predict(
          params, covariates, observations)
      actual_shapes = tree_util.tree_map(
          lambda x: x.mean().shape, distributions_over_observations)
      expected_shapes = tree_util.tree_map(lambda x: x.shape, observations)
      self.assertEqual(actual_shapes, expected_shapes)


if __name__ == '__main__':
  absltest.main()
