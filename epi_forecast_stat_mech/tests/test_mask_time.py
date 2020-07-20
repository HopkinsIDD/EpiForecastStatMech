"""Tests for epi_forecast_stat_mech.mask_time ."""
from absl.testing import absltest

from epi_forecast_stat_mech import mask_time
import numpy as np
import xarray as xr


class TestMaskTest(absltest.TestCase):

  def setUp(self):
    super(TestMaskTest, self).setUp()
    self.num_locations = 5
    self.num_cov = 7
    self.num_time_steps = 3
    self.infections = xr.DataArray(
        np.ones((self.num_locations, self.num_time_steps)),
        dims=['location', 'time'])
    self.cov = xr.DataArray(
        np.zeros((self.num_locations, self.num_time_steps, self.num_cov)),
        dims=['location', 'time', 'dynamic_covariate'])
    self.cov = self.cov.assign_coords({'dynamic_covariate': np.arange(0, self.num_cov).astype(str)})
    data = xr.Dataset()
    data['new_infections'] = self.infections
    data['dynamic_covariates'] = self.cov
    self.data = data

  def test_basic_sanity(self):
    data = self.data.copy(deep=True)
    data.new_infections[dict(time=0)] = np.nan
    tm = mask_time.MaskTime(data)
    mask = tm.mask
    expected_mask = np.ones((self.num_locations, self.num_time_steps))
    expected_mask[:, 0] = 0
    np.testing.assert_array_equal(expected_mask.astype(bool), mask)

  def test_min_value(self):
    min_value = 3
    tm = mask_time.MaskTime(self.data)
    tm.mask_min_infections(min_value=min_value)
    mask = tm.mask
    expected_mask = np.ones((self.num_locations, self.num_time_steps))
    expected_mask[:, :min_value-1] = 0
    np.testing.assert_array_equal(expected_mask.astype(bool), mask)

  def test_dynamic_cov_ones(self):
    cov_name = '1'
    cov_value = 1
    tm = mask_time.MaskTime(self.data)
    tm.mask_dynamic_cov(self.data, cov_name, cov_value)
    mask = tm.mask
    expected_mask = np.zeros((self.num_locations, self.num_time_steps))
    np.testing.assert_array_equal(expected_mask.astype(bool), mask)

  def test_dynamic_cov_zeros(self):
    cov_name = '1'
    cov_value = 0
    tm = mask_time.MaskTime(self.data)
    tm.mask_dynamic_cov(self.data, cov_name=cov_name, mask_value=cov_value)
    mask = tm.mask
    expected_mask = np.ones((self.num_locations, self.num_time_steps))
    np.testing.assert_array_equal(expected_mask.astype(bool), mask)

if __name__ == '__main__':
  absltest.main()
