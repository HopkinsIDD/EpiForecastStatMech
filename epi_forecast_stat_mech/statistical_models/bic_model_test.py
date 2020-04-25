# Lint as: python3
"""Tests for epi_forecast_stat_mech.statistical_models.bic_model."""

from absl.testing import parameterized

from epi_forecast_stat_mech.statistical_models import bic_model
from epi_forecast_stat_mech.statistical_models import linear_model_test

import numpy as np

absl.testing


class BicLinearModelTest(linear_model_test.LinearModelTest):
  """We assert the BIC-wrapped model should pass the tests for `LinearModel`."""

  def get_model(self):
    base_model = super().get_model()
    return bic_model.BICModel(base_model=base_model)


class SoftNonzeroTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(x=np.array([1, 2, 0, 0, 3], dtype=np.float32),
           sharpness=20,
           threshold=.1,
           nonzero_count=3),
      dict(x=np.array([[-1000, 200], [0, 0]], dtype=np.float32),
           sharpness=20,
           threshold=.1,
           nonzero_count=2),
      dict(x=np.array([0, 0, 0, 0], dtype=np.float32),
           sharpness=100,
           threshold=.1,
           nonzero_count=0),
      dict(x=np.arange(1000.),
           sharpness=20,
           threshold=.1,
           nonzero_count=999),
      )
  def testSoftNonzeroAccuracy(self, x, sharpness, threshold, nonzero_count):
    soft_nonzero_count = bic_model.soft_nonzero(x, sharpness, threshold).sum()
    atol = .1 * np.size(x)
    np.testing.assert_allclose(nonzero_count, soft_nonzero_count, atol=atol)


if __name__ == '__main__':
  absltest.main()
