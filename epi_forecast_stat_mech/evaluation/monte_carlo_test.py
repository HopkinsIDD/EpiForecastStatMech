# Lint as: python3
"""Tests for epi_forecast_stat_mech.evaluation.monte_carlo."""

import collections

from absl.testing import absltest
from absl.testing import parameterized

from epi_forecast_stat_mech.evaluation import monte_carlo
from epi_forecast_stat_mech.mechanistic_models import mechanistic_models

import jax
from jax.config import config
import jax.numpy as jnp
import numpy as np

import tensorflow_probability

tfp = tensorflow_probability.experimental.substrates.jax
tfd = tfp.distributions

config.parse_flags_with_absl()  # Necessary for running on TPU.


class DummyEpidemicsRecord(mechanistic_models.EpidemicsRecord):

  @classmethod
  def build(cls, rng, batch_size, time_steps):
    infections = tfd.Poisson(10).sample((batch_size, time_steps), seed=rng)
    cumulative = jnp.cumsum(infections, -1)
    dynamic_covariates = jnp.zeros((batch_size, time_steps, 0))
    t = jnp.broadcast_to(jnp.arange(time_steps), cumulative.shape)
    return cls(
        t=t,
        infections_over_time=infections,
        cumulative_infections=cumulative,
        dynamic_covariates=dynamic_covariates)


class MonteCarloTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(
          rollout_shape=(100,),
          params_shape=([np.array([3, 2])], np.array([3, 12])),
          nsamples=79,
          seed=0),
      dict(
          rollout_shape=(13, 14, 15),
          params_shape=np.array([14, 15]),
          nsamples=23,
          seed=1))
  def testTrajectoriesShape(self, rollout_shape, params_shape, nsamples, seed):

    def rollout_fn(rng, parameters):
      del parameters  # unused
      return jax.random.normal(rng, rollout_shape)

    rng = jax.random.PRNGKey(seed)
    params = jax.tree_map(jnp.ones, params_shape)
    nlocations = jax.tree_leaves(params)[0].shape[0]
    trajectories = monte_carlo.trajectories(rollout_fn, rng, params, nlocations,
                                            nsamples)
    batch_size = jax.tree_leaves(params)[0].shape[0]
    expected_shape = (batch_size, nsamples) + rollout_shape
    self.assertEqual(expected_shape, trajectories.shape)

  @parameterized.parameters(
      dict(
          batch_size=5,
          time_steps=7,
          final_size=7,
          nsamples=4,
          seed=0,
          include_observed=False),
      dict(
          batch_size=4,
          time_steps=17,
          final_size=17 * 2,
          nsamples=3,
          seed=1,
          include_observed=True))
  def testTrajectoriesFromModelShape(self, batch_size, time_steps, final_size,
                                     nsamples, seed, include_observed):
    model = mechanistic_models.ViboudChowellModel()
    rng0, rng1, rng2 = jax.random.split(jax.random.PRNGKey(seed), 3)
    params = tfd.Poisson(30).sample([batch_size, 4], seed=rng0)
    epidemics = DummyEpidemicsRecord.build(rng1, batch_size, time_steps)
    trajectories = monte_carlo.trajectories_from_model(model, params, rng2,
                                                       epidemics, time_steps,
                                                       nsamples,
                                                       include_observed)
    expected_shape = (batch_size, nsamples, final_size)
    self.assertEqual(expected_shape, trajectories.shape)


if __name__ == '__main__':
  absltest.main()
