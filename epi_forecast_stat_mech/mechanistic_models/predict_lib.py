# Lint as: python3
"""Helpers to make predictions using mechanistic models."""

import jax
import jax.numpy as jnp
import numpy as np
import xarray

from epi_forecast_stat_mech.evaluation import monte_carlo  # pylint: disable=g-bad-import-order


def simulate_predictions(mech_model,
                         mech_params,
                         data,
                         epidemics,
                         time_steps,
                         num_samples,
                         rng,
                         include_observed=False):

  predictions = monte_carlo.trajectories_from_model(
      mech_model,
      mech_params,
      rng,
      epidemics,
      time_steps,
      num_samples,
      include_observed)

  # TODO(jamieas): consider indexing by seed.
  sample = np.arange(num_samples)

  # Here we assume evenly spaced integer time values.
  epidemic_time = data.time.data
  try:
    time_delta = epidemic_time[1] - epidemic_time[0]
  except IndexError:  # time 1 may not exist.
    # An alternative is to make it datetime.timedelta(1) for a calendar day.
    time_delta = 1
  time = np.arange(1, time_steps + 1) * time_delta + epidemic_time[-1]
  if include_observed:
    time = np.concatenate((epidemic_time, time))

  location = data.location

  return xarray.DataArray(
      predictions,
      coords=[location, sample, time],
      dims=['location', 'sample', 'time']).rename('new_infections')


def encoded_mech_params_array(data, mech_model, mech_params):
  return xarray.DataArray(
      mech_params,
      dims=('location', 'encoded_param'),
      coords=dict(
          location=data.location,
          encoded_param=list(mech_model.encoded_param_names)))


def mech_params_array(data, mech_model, mech_params):
  decoded_mech_params = jax.vmap(mech_model.decode_params)(mech_params)
  return xarray.DataArray(
      decoded_mech_params,
      dims=('location', 'param'),
      coords=dict(
          location=data.location,
          param=list(mech_model.param_names)))
