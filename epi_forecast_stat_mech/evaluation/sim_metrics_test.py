# Lint as: python3
"""Tests for epi_forecast_stat_mech.evaluation.sim_metrics."""

from absl.testing import absltest
from absl.testing import parameterized

from epi_forecast_stat_mech.evaluation import sim_metrics

import numpy as np
import xarray as xr


class SimMetricsTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(
          sim_metric=sim_metrics.peak_size_error,
          data=xr.DataArray(
              np.zeros((1, 50, 20)), dims=['sample', 'location', 'time']),
          pred=xr.DataArray(
              np.ones((1, 50, 20)), dims=['sample', 'location', 'time']),
          expected_error=np.ones((1, 50))),

      dict(
          sim_metric=sim_metrics.peak_size_error,
          data=xr.DataArray(
              np.array([[[1, 2, 3, 2, 1]]]),
              dims=['sample', 'location', 'time'],
              coords={'time': np.arange(0, 5)}),
          pred=xr.DataArray(
              np.ones((1, 1, 5)),
              dims=['sample', 'location', 'time'],
              coords={'time': np.arange(0, 5)}),
          expected_error=np.array([[-2]])),
      dict(
          sim_metric=sim_metrics.peak_size_error,
          data=xr.DataArray(
              np.array([[[1, 2, 3, 2, 1]]]),
              dims=['sample', 'location', 'time'],
              coords={'time': np.arange(0, 5)}),
          pred=xr.DataArray(
              np.ones((1, 1, 2)),
              dims=['sample', 'location', 'time'],
              coords={'time': np.arange(3, 5)}),
          expected_error=np.array([[0]])),
      dict(
          sim_metric=sim_metrics.peak_size_error,
          data=xr.DataArray(
              np.array([[[1, 2, 3, 2, 1]]]),
              dims=['sample', 'location', 'time'],
              coords={'time': np.arange(0, 5)}),
          pred=xr.DataArray(
              np.ones((1, 1, 4)),
              dims=['sample', 'location', 'time'],
              coords={'time': np.arange(1, 5)}),
          expected_error=np.array([[-2]])),
      dict(
          sim_metric=sim_metrics.epidemic_size_error,
          data=xr.DataArray(
              np.zeros((1, 5, 20)), dims=['sample', 'location', 'time']),
          pred=xr.DataArray(
              np.ones((1, 5, 20)), dims=['sample', 'location', 'time']),
          expected_error=20*np.ones((1, 5))),
  )

  def testSimMetrics(self, pred, data, sim_metric, expected_error):
    peak_error = sim_metric(data, pred)
    np.testing.assert_array_equal(peak_error.data, expected_error)

if __name__ == '__main__':
  absltest.main()
