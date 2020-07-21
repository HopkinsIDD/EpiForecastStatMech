"""Tests for epi_forecast_stat_mech.mechanistic_models.observables."""

from absl.testing import absltest
from absl.testing import parameterized
from epi_forecast_stat_mech.mechanistic_models import mechanistic_models
from epi_forecast_stat_mech.mechanistic_models import observables
import numpy as np
from jax import numpy as jnp
from jax.config import config


config.parse_flags_with_absl()  # Necessary for running on TPU.


class ObservablesTest(parameterized.TestCase):
  # jnp's can't be in arguments, so we cast the expected result after the fact.
  @parameterized.parameters(
      dict(
          observable=observables.ObserveSpecified(['log_K']),
          mech_model=mechanistic_models.ViboudChowellModel(),
          mech_params=np.asarray([1., 1., 1., 10.], dtype=np.float32),
          epidemic=None,
          expected_result={'log_K': np.float32([10.])}),
      dict(
          observable=observables.InternalParams(),
          mech_model=mechanistic_models.ViboudChowellModel(),
          mech_params=np.asarray([1., 1., 1., 10.], dtype=np.float32),
          epidemic=None,
          expected_result={'log_r': np.float32([1.]),
                           'log_a': np.float32([1.]),
                           'log_p': np.float32([1.]),
                           'log_K': np.float32([10.])}),
  )
  def test_expected(self, observable, mech_model, mech_params, epidemic,
                    expected_result):
    jnp_expected_result = {
        key: jnp.asarray(value) for key, value in expected_result.items()
    }
    result = observable.observables(mech_model, mech_params, epidemic)
    self.assertSameStructure(result, jnp_expected_result)
    for key, value in result.items():
      expected_value = jnp_expected_result[key]
      self.assertAlmostEqual(np.asarray(value), expected_value, places=3)


if __name__ == '__main__':
  absltest.main()
