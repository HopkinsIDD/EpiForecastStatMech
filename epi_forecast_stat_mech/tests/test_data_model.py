"""Tests for epi_forecast_stat_mech/data_model ."""
from absl.testing import absltest

from epi_forecast_stat_mech import data_model
import numpy as np
import xarray as xr


class TestNewDataArray(absltest.TestCase):

  def setUp(self):
    super(TestNewDataArray, self).setUp()
    self.num_locations = 43
    self.num_static_covariates = 2

  def test_empty_dataarray(self):
    trajectories = data_model.new_dataarray({'location': self.num_locations})
    assert trajectories.data.shape[0] == self.num_locations
    assert len(trajectories.dims) == 1
    assert trajectories.dims[0] == 'location'

  def test_reassign_dataarray(self):
    new_covariates = np.array([10., 20.])
    trajectories = data_model.new_dataarray(
        {'static_covariate': self.num_static_covariates})
    trajectories.data = new_covariates
    assert trajectories.data.shape[0] == self.num_static_covariates
    np.testing.assert_array_equal(trajectories.data, new_covariates)


class TestNewModel(absltest.TestCase):

  def setUp(self):
    super(TestNewModel, self).setUp()
    self.num_locations = 43
    self.num_time_steps = 77
    self.num_static_covariates = 2

  def test_empty_model(self):
    trajectories = data_model.new_model(self.num_locations, self.num_time_steps,
                                        self.num_static_covariates)
    np.testing.assert_array_equal(
        trajectories.new_infections.data,
        np.zeros((self.num_locations, self.num_time_steps)))
    np.testing.assert_array_equal(
        trajectories.static_covariates.data,
        np.zeros((self.num_locations, self.num_static_covariates)))

    trajectories = data_model.new_model(self.num_locations, self.num_time_steps,
                                        self.num_static_covariates)
    np.testing.assert_array_equal(
        trajectories.new_infections.data,
        np.zeros((self.num_locations, self.num_time_steps)))
    np.testing.assert_array_equal(
        trajectories.static_covariates.data,
        np.zeros((self.num_locations, self.num_static_covariates)))


class TestSums(absltest.TestCase):

  def setUp(self):
    super(TestSums, self).setUp()
    self.num_locations = 2
    self.num_time_steps = 4
    self.num_static_covariates = 2
    self.ds = data_model.new_model(self.num_locations, self.num_time_steps,
                                   self.num_static_covariates)
    self.ds.new_infections.data = np.array([[1, 2, 3, 4], [0, 0, 0, 0]])

  def test_cumsum(self):
    new_da = data_model.calculate_cumulative_infections(self.ds.new_infections)
    np.testing.assert_array_equal(new_da.data,
                                  np.array([[1, 3, 6, 10], [0, 0, 0, 0]]))

  def test_sums(self):
    new_da = data_model.calculate_total_infections(self.ds.new_infections)
    np.testing.assert_array_equal(new_da.data, np.array([10, 0]))

  def test_all_sums(self):
    new_ds = data_model.calculate_infection_sums(self.ds)
    np.testing.assert_array_equal(new_ds.cumulative_infections.data,
                                  np.array([[1, 3, 6, 10], [0, 0, 0, 0]]))
    np.testing.assert_array_equal(new_ds.total_infections.data,
                                  np.array([10, 0]))


class TestShiftData(absltest.TestCase):

  def setUp(self):
    super(TestShiftData, self).setUp()
    self.num_locations = 3
    self.num_time_steps = 5
    self.dataarray = xr.DataArray(
        np.ones((self.num_locations, self.num_time_steps)),
        dims=['location', 'time'])
    self.shifts = 2 * np.ones(self.num_locations, dtype=int)

  def test_shift_array(self):
    shifted_data, _ = data_model._helper_shift_dataarray(
        self.shifts, self.dataarray)
    expected_num_shifts = 2 * self.num_locations
    expected_sum = self.num_locations * self.num_time_steps - expected_num_shifts
    assert expected_sum == shifted_data.sum()


class TestValidateData(absltest.TestCase):

  def setUp(self):
    super(TestValidateData, self).setUp()
    self.num_locations = 43
    self.num_time_steps = 77
    self.num_static_covariates = 2
    self.data = data_model.new_model(self.num_locations, self.num_time_steps,
                                     self.num_static_covariates)
    data_model.set_sensible_canonical_split_time(self.data)

  def test_validate_ok(self):
    data_model.validate_data(self.data)

  def test_validate_ok2(self):
    data_model.validate_data(self.data)

  def test_validate_ok_no_samples(self):
    data_model.validate_data(self.data, require_no_samples=True)

  def test_type_failure(self):
    with self.assertRaisesRegex(ValueError, 'data must be an xarray'):
      data_model.validate_data(None)

  def test_no_static(self):
    data = self.data.copy()
    del data['static_covariates']
    with self.assertRaisesRegex(ValueError,
                                'data is missing required data_vars:.*'):
      data_model.validate_data(data)

  def test_no_missing_in_static(self):
    data = self.data.copy()
    data.static_covariates[1, 1] = np.NaN
    with self.assertRaisesRegex(
        ValueError, 'null data is not allowed in data.static_covariates.'):
      data_model.validate_data(data)


class TestIntegerTimes(absltest.TestCase):

  def setUp(self):
    num_locations = 2
    num_time_steps = 4
    num_static_covariates = 2
    self.ds = data_model.new_model(num_locations, num_time_steps,
                                   num_static_covariates)
    self.ds.new_infections.data = np.array([[0, 1, 2, 3], [0, 0, 0, 0]])
    self.ds['time'] = self.ds.time * np.timedelta64(
        1, 'D') + np.datetime64('2020-03-04')

  def test_convert_to_integer_time(self):
    # Default behavior should match day-of-year for dates in 2020 (e.g. Jan 1 is
    # day 1, Feb 1 is day 32, etc.)
    ds = data_model.convert_data_to_integer_time(self.ds)
    assert (ds.time == self.ds.time.dt.dayofyear).all()

    ds = data_model.convert_data_to_integer_time(
        self.ds, method='days_from_first_case')
    assert ds.time[0].item() == -1, ds.time

    ds = data_model.convert_data_to_integer_time(self.ds, method='numpy_index')
    assert ds.time[0].item() == 0, ds.time

  def test_datetime_int_conversion_with_nonnone_ds(self):
    ds = data_model.convert_data_to_integer_time(
        self.ds, method='days_from_first_case')
    np.testing.assert_array_equal(
        data_model.int_to_datetime(ds.time, ds), ds.original_time)
    np.testing.assert_array_equal(
        data_model.datetime_to_int(ds.original_time, ds), ds.time)


if __name__ == '__main__':
  absltest.main()
