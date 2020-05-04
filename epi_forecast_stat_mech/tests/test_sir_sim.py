"""Tests for epi_forecast_stat_mech/sir_sim ."""
from absl.testing import absltest

from epi_forecast_stat_mech import sir_sim
import numpy as np


class TestGenerateSirSimulations(absltest.TestCase):

  def setUp(self):
    super(TestGenerateSirSimulations, self).setUp()
    self.num_simulations = 1
    self.num_epidemics = 50
    self.num_important_cov = 1
    self.num_unimportant_cov = 2
    self.num_time_steps = 500

  def test_basic_sanity(self):
    trajectories = sir_sim.generate_simulations(
        sir_sim.generate_betas_many_cov2,
        (self.num_epidemics, self.num_important_cov, self.num_unimportant_cov),
        self.num_simulations,
        self.num_epidemics,
        self.num_time_steps,
        constant_pop_size=10000)
    assert trajectories.new_infections.shape == (self.num_simulations,
                                                 self.num_epidemics,
                                                 self.num_time_steps)

  def test_small_time_steps(self):
    with self.assertRaisesRegex(ValueError, '.*'):
      trajectories = sir_sim.generate_simulations(
          sir_sim.generate_betas_many_cov2,
          (self.num_epidemics, self.num_important_cov,
           self.num_unimportant_cov),
          self.num_simulations,
          self.num_epidemics,
          num_time_steps=20,
          constant_pop_size=10000)


if __name__ == '__main__':
  absltest.main()
