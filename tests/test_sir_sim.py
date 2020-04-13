from absl.testing import absltest

from epi_forecast_stat_mech import sir_sim


class GenerateSirSimulations(absltest.TestCase):

  def test_basic_sanity(self):
    num_simulations = 1
    num_epidemics = 50
    num_important_cov = 1
    num_unimportant_cov = 2
    trajectories = sir_sim.generate_SIR_simulations(
        sir_sim.generate_betas_many_cov2,
        (num_epidemics, num_important_cov, num_unimportant_cov),
        num_simulations, num_epidemics, constant_pop_size=10000)
    assert len(trajectories) == num_epidemics
