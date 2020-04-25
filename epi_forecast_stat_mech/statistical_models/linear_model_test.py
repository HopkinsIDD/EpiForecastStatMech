# Lint as: python3
"""Tests for epi_forecast_stat_mech.statistical_models.linear_model."""

from absl.testing import parameterized
from epi_forecast_stat_mech.statistical_models import linear_model

from epi_forecast_stat_mech.statistical_models import tree_util
import jax
import jax.numpy as jnp

from absl.testing import absltest


class LinearModelTest(parameterized.TestCase):

  def assertBroadcastableTo(self, x, shape):
    try:
      jnp.broadcast_to(x, shape)
    except ValueError:
      self.fail(f'The value `x` with shape {x.shape} '
                f'was not broadcastable to shape {shape}.')

  def get_model(self):
    return linear_model.LinearModel()

  @parameterized.parameters(
      dict(covariate_shapes=(5, 5, 5),
           mech_param_shapes=(2, 3, 4),
           num_epidemics=20),
      dict(covariate_shapes=(1, 2, 3, 4, 5),
           mech_param_shapes=(100,),
           num_epidemics=1000),
      )
  def testShapes(
      self, covariate_shapes, mech_param_shapes, num_epidemics):
    covariate_dim = sum(covariate_shapes)
    prediction_dim = sum(mech_param_shapes)
    params = linear_model.LinearParameters.init(covariate_dim,
                                                prediction_dim)
    covariates = jax.tree_map(lambda c: jnp.zeros([num_epidemics, c]),
                              covariate_shapes)
    mech_params = jax.tree_map(lambda c: jnp.zeros([num_epidemics, c]),
                               mech_param_shapes)

    self.assertEqual(params.alpha.shape, (covariate_dim, prediction_dim))
    self.assertEqual(params.intercept.shape, (1, prediction_dim))

    model = self.get_model()

    # Prior should be broadcastable to the shape of `params`.
    log_prior = model.log_prior(params)
    packed_params, _ = tree_util.pack(params, axis=0)
    self.assertBroadcastableTo(log_prior, packed_params.shape)

    # Likelihood should be broadcastable to the shape of `mech_params`.
    log_likelihood = model.log_likelihood(params, covariates, mech_params)
    packed_mech_params, _ = tree_util.pack(mech_params)
    self.assertBroadcastableTo(log_likelihood, packed_mech_params.shape)

    prediction = model.predict(params, covariates)
    self.assertBroadcastableTo(prediction, packed_mech_params.shape)


if __name__ == '__main__':
  absltest.main()
