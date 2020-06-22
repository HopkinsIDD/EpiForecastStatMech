# Lint as: python3
"""Tests for epi_forecast_stat_mech.datasets.nanhandling."""
from absl.testing import absltest
from epi_forecast_stat_mech.datasets import nanhandling
import numpy as np


class NanhandlingTest(absltest.TestCase):

  def setUp(self):
    super(NanhandlingTest, self).setUp()
    self.array = np.array([[5, np.nan, np.nan, 7, 2], [3, np.nan, 1, 8, np.nan],
                           [np.nan, np.nan, np.nan, 3, 1],
                           [np.nan, np.nan, np.nan, np.nan, np.nan],
                           [np.nan, np.nan, np.nan, 3, np.nan]])

  def test_fillna_zero(self):
    result = nanhandling.fillna_zero(self.array)
    expected = np.array([[5, 0, 0, 7, 2], [3, 0, 1, 8, 0], [0, 0, 0, 3, 1],
                         [0, 0, 0, 0, 0], [0, 0, 0, 3, 0]])
    self.assertTrue(np.array_equal(expected, result))

  def test_median_filter(self):
    result = nanhandling.median_filter(self.array)
    expected = np.array([[5, 5, 5, 2, 2], [3, 3, 3, 4.5, 4.5],
                         [np.nan, 3, 2, 1, 1],
                         [np.nan, np.nan, np.nan, np.nan, np.nan],
                         [np.nan, 3, 3, 3, 3]])
    np.testing.assert_array_equal(result, expected)

  def test_mean_filter(self):
    result = nanhandling.mean_filter(self.array)
    expected = np.array(
        [[
            5, (5 + 5 + 7) / 3, (5 + 7 + 2) / 3, (7 + 2 + 2) / 3,
            (7 + 2 + 2 + 2) / 4
        ],
         [(3 + 3 + 3 + 1) / 4, (3 + 3 + 1 + 8) / 4, (3 + 1 + 8) / 3,
          (1 + 8) / 2, (1 + 8) / 2],
         [np.nan, 3, (3 + 1) / 2, (3 + 1 + 1) / 3, (3 + 1 + 1 + 1) / 4],
         [np.nan, np.nan, np.nan, np.nan, np.nan], [np.nan, 3, 3, 3, 3]])
    np.testing.assert_array_equal(result, expected)

  def test_fillna_beginning(self):
    result = nanhandling.fillna_beginning(self.array)
    expected = np.array([[5, np.nan, np.nan, 7, 2],
                         [3, np.nan, 1, 8, np.nan],
                         [0, 0, 0, 3, 1],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 3, np.nan]])
    np.testing.assert_array_equal(result, expected)

  def test_fillna_ffill(self):
    result = nanhandling.fillna_ffill(self.array)
    expected = np.array([[5, 5, 5, 7, 2],
                         [3, 3, 1, 8, 8],
                         [np.nan, np.nan, np.nan, 3, 1],
                         [np.nan, np.nan, np.nan, np.nan, np.nan],
                         [np.nan, np.nan, np.nan, 3, 3]])
    np.testing.assert_array_equal(result, expected)

  def test_fillna_bfill(self):
    result = nanhandling.fillna_bfill(self.array)
    expected = np.array([[5, 7, 7, 7, 2],
                         [3, 1, 1, 8, np.nan],
                         [3, 3, 3, 3, 1],
                         [np.nan, np.nan, np.nan, np.nan, np.nan],
                         [3, 3, 3, 3, np.nan]])
    np.testing.assert_array_equal(result, expected)

  def test_fillna_interp(self):
    result = nanhandling.fillna_interp(self.array)
    expected = np.array([[5, 5 + (2/3), 5+ (4/3), 7, 2],
                         [3, 2, 1, 8, np.nan],
                         [np.nan, np.nan, np.nan, 3, 1],
                         [np.nan, np.nan, np.nan, np.nan, np.nan],
                         [np.nan, np.nan, np.nan, 3, np.nan]])
    np.testing.assert_array_equal(result, expected)

  def test_longest_nans(self):
    result = nanhandling.longest_nans(self.array)
    expected = np.array([2, 1, 3, 5, 3])
    np.testing.assert_array_equal(result, expected)

if __name__ == '__main__':
  absltest.main()
