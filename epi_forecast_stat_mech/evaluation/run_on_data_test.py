# Lint as: python3
"""Tests for epi_forecast_stat_mech.evaluation.run_on_data."""

from absl.testing import absltest


from epi_forecast_stat_mech import high_level
from epi_forecast_stat_mech import sir_sim
from epi_forecast_stat_mech.evaluation import run_on_data
from epi_forecast_stat_mech.evaluation import sim_metrics
import functools
import numpy as np


def create_synthetic_dataset(
    seed=0,
    num_epidemics=50,
    num_important_cov=1,
    num_unimportant_cov=2,
    constant_pop_size=200000,
    num_time_steps=100,
):
  """Creates synthetic data."""
  np.random.seed(seed)  # TODO(shoyer): use np.random.RandomState
  num_simulations = 1
  trajectories = sir_sim.generate_simulations(
      sir_sim.generate_betas_many_cov2,
      (num_epidemics, num_important_cov, num_unimportant_cov),
      num_simulations, num_epidemics,
      constant_pop_size=constant_pop_size,
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

  def test_TrainModel(self):
    # very similar to test_high_level
    split_args = (20,)
    split_function = run_on_data.train_test_split_time
    data = create_synthetic_dataset(num_epidemics=50, num_time_steps=100)
    data = data.squeeze('sample')
    estimator = high_level.StatMechEstimator()
    predictions = run_on_data.train_model(
        data, estimator, split_function, split_args)
    self.assertCountEqual(['location', 'sample', 'time'], predictions.dims)
    self.assertLen(predictions.time, 100-split_args[0])
    np.testing.assert_array_equal(data.location, predictions.location)

  def test_EvaluateModel(self):
    predictions = create_synthetic_dataset(num_epidemics=50, num_time_steps=100)
    data = create_synthetic_dataset(num_epidemics=50, num_time_steps=100)
    partial_func = functools.partial(sim_metrics.peak_size_error,
                                     data.new_infections)
    metric_dict = {'peak': partial_func}
    eval_data = run_on_data.evaluate_model(predictions, metric_dict)
    self.assertLen(eval_data.time, 100)


if __name__ == '__main__':
  absltest.main()
