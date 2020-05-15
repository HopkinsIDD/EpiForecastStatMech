"""Tests for epi_forecast_stat_mech/sir_sim ."""
import functools
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
    beta_fn = functools.partial(sir_sim.generate_betas_many_cov2,
                                num_pred=self.num_important_cov,
                                num_not_pred=self.num_unimportant_cov)
    trajectories = sir_sim.generate_simulations(
        beta_fn,
        self.num_simulations,
        self.num_epidemics,
        self.num_time_steps,
        constant_pop_size=10000)
    assert trajectories.new_infections.shape == (self.num_simulations,
                                                 self.num_epidemics,
                                                 self.num_time_steps)

  def test_small_time_steps(self):
    with self.assertRaisesRegex(ValueError, '.*'):
      beta_fn = functools.partial(sir_sim.generate_betas_many_cov2,
                                  num_pred=self.num_important_cov,
                                  num_not_pred=self.num_unimportant_cov)
      trajectories = sir_sim.generate_simulations(
          beta_fn,
          self.num_simulations,
          self.num_epidemics,
          num_time_steps=20,
          constant_pop_size=10000)

  def test_random_dynamic_cov(self):
    """Test we generate growth rates that change at a random time."""
    static_beta_fn = functools.partial(sir_sim.generate_betas_many_cov2,
                                       num_pred=self.num_important_cov,
                                       num_not_pred=self.num_unimportant_cov)
    dynamic_beta_fn = sir_sim.gen_dynamic_beta_random_time
    trajectories = sir_sim.generate_simulations(
        static_beta_fn,
        self.num_simulations,
        self.num_epidemics,
        self.num_time_steps,
        constant_pop_size=10000,
        gen_dynamic_beta_fn=dynamic_beta_fn)

    shift_growth_rate = trajectories.growth_rate.shift(time=1)
    num_diff_betas = (trajectories.growth_rate[:, :-1] !=
                      shift_growth_rate[:, 1:]).sum()
    assert num_diff_betas == self.num_epidemics

  def test_social_distancing_sanity(self):
    beta_fn = functools.partial(sir_sim.generate_betas_many_cov2,
                                num_pred=self.num_important_cov,
                                num_not_pred=self.num_unimportant_cov)
    trajectories = sir_sim.generate_social_distancing_simulations(
        beta_fn,
        sir_sim.gen_social_distancing_weight,
        self.num_simulations,
        self.num_epidemics,
        self.num_time_steps,
        constant_pop_size=10000)
    assert trajectories.new_infections.shape == (self.num_simulations,
                                                 self.num_epidemics,
                                                 self.num_time_steps)

if __name__ == '__main__':
  absltest.main()
