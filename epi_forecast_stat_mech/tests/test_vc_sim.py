"""Tests for epi_forecast_stat_mech/vc_sim ."""
from absl.testing import absltest

from epi_forecast_stat_mech import data_model
from epi_forecast_stat_mech import vc_sim


class TestGenerateVCSimulations(absltest.TestCase):

  def setUp(self):
    super(TestGenerateVCSimulations, self).setUp()
    self.num_samples = 1
    self.num_locations = 50
    self.num_time_steps = 500

  def test_basic_sanity(self):
    trajectories = vc_sim.generate_simulations(
        vc_sim.final_size_poisson_dist,
        self.num_samples,
        self.num_locations,
        self.num_time_steps)
    data_model.validate_data(trajectories)
    assert trajectories.new_infections.shape == (self.num_samples,
                                                 self.num_locations,
                                                 self.num_time_steps)


if __name__ == '__main__':
  absltest.main()
