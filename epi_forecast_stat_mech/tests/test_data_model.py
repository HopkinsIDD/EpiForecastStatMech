"""Tests for epi_forecast_stat_mech/data_model ."""
from absl.testing import absltest

from epi_forecast_stat_mech import data_model
import numpy as np


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
    self.num_samples = 1
    self.num_locations = 43
    self.num_time_steps = 77
    self.num_static_covariates = 2

  def test_empty_model(self):
    trajectories = data_model.new_model(self.num_samples, self.num_locations,
                                        self.num_time_steps,
                                        self.num_static_covariates)
    np.testing.assert_array_equal(
        trajectories.new_infections.data,
        np.zeros((self.num_samples, self.num_locations, self.num_time_steps)))
    np.testing.assert_array_equal(
        trajectories.static_covariates.data,
        np.zeros((self.num_locations, self.num_static_covariates)))

    trajectories = data_model.new_model(self.num_samples, self.num_locations,
                                        self.num_time_steps,
                                        self.num_static_covariates)
    np.testing.assert_array_equal(
        trajectories.new_infections.data,
        np.zeros((self.num_samples, self.num_locations, self.num_time_steps)))
    np.testing.assert_array_equal(
        trajectories.static_covariates.data,
        np.zeros((self.num_locations, self.num_static_covariates)))


class TestSums(absltest.TestCase):

  def setUp(self):
    super(TestSums, self).setUp()
    self.num_samples = 1
    self.num_locations = 2
    self.num_time_steps = 4
    self.num_static_covariates = 2
    self.ds = data_model.new_model(self.num_samples, self.num_locations,
                                   self.num_time_steps,
                                   self.num_static_covariates)
    self.ds.new_infections.data = np.array([[[1, 2, 3, 4], [0, 0, 0, 0]]])

  def test_cumsum(self):
    new_da = data_model.calculate_cumulative_infections(self.ds.new_infections)
    np.testing.assert_array_equal(
        new_da.data,
        np.array([[[1, 3, 6, 10], [0, 0, 0, 0]]]))

  def test_sums(self):
    new_da = data_model.calculate_total_infections(self.ds.new_infections)
    np.testing.assert_array_equal(
        new_da.data,
        np.array([[10, 0]]))

  def test_all_sums(self):
    new_ds = data_model.calculate_infection_sums(self.ds)
    np.testing.assert_array_equal(
        new_ds.cumulative_infections.data,
        np.array([[[1, 3, 6, 10], [0, 0, 0, 0]]]))
    np.testing.assert_array_equal(
        new_ds.total_infections.data,
        np.array([[10, 0]]))

if __name__ == '__main__':
  absltest.main()

