# Lint as: python3
"""Tests for epi_forecast_stat_mech.evaluation.sim_metrics."""

from absl.testing import absltest
from absl.testing import parameterized

from epi_forecast_stat_mech.evaluation import sim_metrics

import numpy as np
import pandas as pd
import xarray as xr


class SimMetricsHelperTest(parameterized.TestCase):
  # TODO(edklein, mcoram): add tests with NaNs
  @parameterized.parameters(
      dict(
          sim_fun=sim_metrics._helper_data_validator,
          data=xr.DataArray(
              data=np.array([1, 2, 3, 2, 1]),
              dims=['time'],
              coords={'time': np.arange(0, 5)}),
          pred=xr.DataArray(
              data=np.ones(2),
              dims=['time'],
              coords={'time': np.arange(3, 5)}),
          expected_result=np.array([1, 2, 3, 1, 1])),
      dict(
          sim_fun=sim_metrics._helper_data_validator,
          data=xr.DataArray(
              data=np.array([1, 2, 3, 2, 1]),
              dims=['time'],
              coords={
                  'time': pd.date_range('2000-01-01', periods=5),
              }),
          pred=xr.DataArray(
              data=np.array([10, 11]),
              dims=['time'],
              coords={'time': pd.date_range('2000-01-04', periods=2)[::-1]}),
          expected_result=np.array([1, 2, 3, 10, 11])),
      dict(
          sim_fun=sim_metrics._helper_data_validator,
          data=xr.DataArray(
              data=np.array([[[1, 2, 3, 2, 1],
                              [1, 3, 2, 4, 1],
                              [1, 4, 1, 1, 1]],
                             [[10, 14, 12, 11, 10],
                              [11, 12, 13, 14, 15],
                              [20, 21, 22, 24, 25]]]),
              dims=['location', 'sample', 'time'],
              coords={
                  'time': pd.date_range('2000-01-01', periods=5),
                  'location': np.array(['A', 'B']),
                  'sample': np.array([0, 1, 2])}),
          pred=xr.DataArray(
              data=np.array([[[100, 101], [101, 102], [103, 104]],
                             [[105, 106], [107, 108], [109, 110]]]),
              dims=['location', 'sample', 'time'],
              coords={
                  'time': pd.date_range('2000-01-04', periods=2),
                  'location': np.array(['A', 'B']),
                  'sample': np.array([0, 1, 2])
              }),
          expected_result=np.array([[[1, 2, 3, 100, 101],
                                     [1, 3, 2, 101, 102],
                                     [1, 4, 1, 103, 104]],
                                    [[10, 14, 12, 105, 106],
                                     [11, 12, 13, 107, 108],
                                     [20, 21, 22, 109, 110]]])),
      dict(
          sim_fun=sim_metrics._helper_data_validator,
          data=xr.DataArray(
              data=np.array([1, 2, 3, 2, 1]),
              dims=['time'],
              coords={'time': np.arange(0, 5)}),
          pred=xr.DataArray(
              data=np.ones((2, 2)),
              dims=['model', 'time'],
              coords={'time': np.arange(3, 5)}),
          expected_result=np.array([[1, 2, 3, 1, 1], [1, 2, 3, 1, 1]])),

      dict(
          sim_fun=sim_metrics._helper_data_validator,
          data=xr.DataArray(
              data=np.array([1, 2, 3, 2]),
              dims=['time'],
              coords={
                  'time': pd.date_range('2000-01-01', periods=4),
              }),
          pred=xr.DataArray(
              data=np.array([10, 11]),
              dims=['time'],
              coords={'time': pd.date_range('2000-01-04', periods=2)}),
          expected_result=np.array([1, 2, 3, 10])),
      )
  def testSimMetricsHelpers(self, data, pred, sim_fun, expected_result):
    result = sim_fun(data, pred)
    np.testing.assert_array_equal(result, expected_result)

  @parameterized.parameters(
      dict(
          sim_fun=sim_metrics._helper_data_validator,
          data=xr.DataArray(
              data=np.array([1, 2, 3, 2, 1]),
              dims=['time'],
              coords={
                  'time': pd.date_range('2000-01-01', periods=5),
              }),
          pred=xr.DataArray(
              data=np.ones((2)),
              dims=['time'],
              coords={'time': pd.date_range('2000-01-01', periods=2)}),
          expected_result=-1
          ),
      dict(
          sim_fun=sim_metrics._helper_data_validator,
          data=xr.DataArray(
              data=np.array([1, 2, 3, 2, 1]),
              dims=['time'],
              coords={
                  'time': pd.date_range('2000-01-01', periods=5),
              }),
          pred=xr.DataArray(
              data=np.ones((4)),
              dims=['time'],
              coords={'time': pd.date_range('2000-01-01', periods=4)}),
          expected_result=-1
          )
      )
  def testExceptions(self, data, pred, sim_fun, expected_result):
    with self.assertRaises(ValueError):
      sim_fun(data, pred)


class SimMetricsEvalTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(
          sim_metric=sim_metrics.peak_size_error,
          data=xr.DataArray(
              data=np.zeros((1, 50, 20)),
              dims=['sample', 'location', 'time'],
              coords={'time': np.arange(0, 20)}),
          pred=xr.DataArray(
              data=np.ones((1, 50, 20)),
              dims=['sample', 'location', 'time'],
              coords={'time': np.arange(0, 20)}),
          expected_error=-np.ones((1, 50))),
      dict(
          sim_metric=sim_metrics.peak_size_error,
          data=xr.DataArray(
              data=np.array([1, 2, 3, 2, 1]),
              dims=['time'],
              coords={'time': np.arange(0, 5)}),
          pred=xr.DataArray(
              data=np.ones(5),
              dims=['time'],
              coords={'time': np.arange(0, 5)}),
          expected_error=2),
      dict(
          sim_metric=sim_metrics.peak_size_error,
          data=xr.DataArray(
              data=np.array([1, 2, 3, 2, 1]),
              dims=['time'],
              coords={'time': np.arange(0, 5)}),
          pred=xr.DataArray(
              data=np.ones(2),
              dims=['time'],
              coords={'time': np.arange(3, 5)}),
          expected_error=0),
      dict(
          sim_metric=sim_metrics.peak_size_error,
          data=xr.DataArray(
              data=np.array([1, 2, 3, 2, 1]),
              dims=['time'],
              coords={'time': np.arange(0, 5)}),
          pred=xr.DataArray(
              data=np.ones((4)),
              dims=['time'],
              coords={'time': np.arange(1, 5)}),
          expected_error=2),
      dict(
          sim_metric=sim_metrics.peak_time_error,
          data=xr.DataArray(
              data=np.array([1, 2, 3, 2, 1]),
              dims=['time'],
              coords={'time': np.arange(0, 5)}),
          pred=xr.DataArray(
              data=np.ones(4),
              dims=['time'],
              coords={'time': np.arange(1, 5)}),
          expected_error=2),
      dict(
          sim_metric=sim_metrics.peak_time_error,
          data=xr.DataArray(
              data=np.array([1, 2, 3, 2, 1]),
              dims=['time'],
              coords={'time': np.arange(0, 5)}),
          pred=xr.DataArray(
              data=np.ones(2),
              dims=['time'],
              coords={'time': np.arange(3, 5)}),
          expected_error=0),
      dict(
          sim_metric=sim_metrics.total_size_error,
          data=xr.DataArray(
              data=np.zeros((1, 5, 20)),
              dims=['sample', 'location', 'time'],
              coords={'time': np.arange(0, 20)}),
          pred=xr.DataArray(
              data=np.ones((1, 5, 20)),
              dims=['sample', 'location', 'time'],
              coords={'time': np.arange(0, 20)}),
          expected_error=-20 * np.ones((1, 5))),
      dict(
          sim_metric=sim_metrics.time_percent_complete_error,
          data=xr.DataArray(
              data=np.array([0, 1, 1, 1, 1]),
              dims=['time'],
              coords={'time': np.arange(0, 5)}),
          pred=xr.DataArray(
              data=np.array([0, 0, 2, 2, 0]),
              dims=['time'],
              coords={'time': np.arange(0, 5)}),
          expected_error=1),
      dict(
          sim_metric=sim_metrics.time_percent_complete_error,
          data=xr.DataArray(
              data=np.array([0, 1, 1, 1, 1]),
              dims=['time'],
              coords={'time': np.arange(0, 5)}),
          pred=xr.DataArray(
              data=np.array([[0, 0, 2, 2, 12]]),
              dims=['metric', 'time'],
              coords={'time': np.arange(0, 5)}),
          expected_error=np.array([-2])),
  )
  def testSimMetrics(self, data, pred, sim_metric, expected_error):
    values = sim_metric(data, pred)
    error = values.sel(value_type='difference')
    np.testing.assert_array_equal(error, expected_error)


if __name__ == '__main__':
  absltest.main()
