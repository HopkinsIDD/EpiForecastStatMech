# Lint as: python3
"""Helpers to make predictions using mechanistic models."""

import numpy as np
import xarray

from epi_forecast_stat_mech.evaluation import monte_carlo  # pylint: disable=g-bad-import-order


def simulate_predictions(mech_model, mech_params, data, epidemics, time_steps,
                         num_samples, rng):

  predictions = monte_carlo.trajectories_from_model(mech_model, mech_params,
                                                    rng, epidemics, time_steps,
                                                    num_samples)

  # TODO(jamieas): consider indexing by seed.
  sample = np.arange(num_samples)

  # Here we assume evenly spaced integer time values.
  epidemic_time = data.time.data
  time_delta = epidemic_time[1] - epidemic_time[0]
  time = np.arange(1, time_steps + 1) * time_delta + epidemic_time[-1]

  location = data.location

  return xarray.DataArray(
      predictions,
      coords=[location, sample, time],
      dims=["location", "sample", "time"]).rename("new_infections")
