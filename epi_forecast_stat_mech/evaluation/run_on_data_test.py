# Lint as: python3
"""Tests for epi_forecast_stat_mech.evaluation.run_on_data."""
import functools

from absl.testing import absltest

from epi_forecast_stat_mech import high_level
from epi_forecast_stat_mech import sir_sim
from epi_forecast_stat_mech.evaluation import run_on_data
from epi_forecast_stat_mech.evaluation import sim_metrics
import numpy as np


def create_synthetic_dataset(
    seed=0,
    num_epidemics=50,
    num_important_cov=1,
    num_unimportant_cov=2,
    num_time_steps=100,
):
  """Creates synthetic data."""
  np.random.seed(seed)  # TODO(shoyer): use np.random.RandomState
  num_simulations = 1
  beta_fn = functools.partial(sir_sim.generate_betas_many_cov2,
                              num_pred=num_important_cov,
                              num_not_pred=num_unimportant_cov)
  trajectories = sir_sim.generate_simulations(
      beta_fn,
      num_simulations, num_epidemics,
      num_time_steps=num_time_steps)
  return trajectories


class TestRunOnData(absltest.TestCase):
  """Tests for run_on_data."""

  def test_TrainTestSplitTime(self):
    """Verify we can split data at a time point."""
    split_day = 20

    data = create_synthetic_dataset(num_epidemics=50, num_time_steps=100)
    train_data, test_data = run_on_data.train_test_split_time(data, split_day)

    self.assertCountEqual(
        ['location', 'time', 'static_covariate'],
        test_data.dims)
    self.assertLen(test_data.time, 100 - split_day)
    self.assertLen(train_data.time, split_day)
    np.testing.assert_array_equal(data.location, train_data.location)
    np.testing.assert_array_equal(data.location, test_data.location)

if __name__ == '__main__':
  absltest.main()
