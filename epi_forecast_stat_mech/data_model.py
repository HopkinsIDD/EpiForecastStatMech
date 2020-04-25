# Lint as: python3
"""Data model for representing real data or simulated data.

Define an xarray.DataSeries to hold new_infection trajectories over time
and associated covariates. This is our data model that will be used for
'real' and simulated data. Define helper functions to calculate useful
secondary quantities and store them in the DataSeries.
"""
from .constants import coord_units
from .constants import coordinates

import numpy as np
import xarray as xr

valid_dimensions = ['sample', 'location', 'time', 'static_covariate',
                    'dynamic_covariate']


def new_dataarray(parameters):
  """Helper function to create xr.DataArrays.

  Create xr.DataArrays with consistent dimensions, coordinates, and units.

  Args:
    parameters: a dictionary of (dimension: number) pairs.
      Dimension must be in valid_dimensions. Number is the length of that axis.
  Returns:
    da: a xr.DataArray containing zeros along all specified dimensions
  """
  # Check all dimensions are valid
  for dim in parameters.keys():
    if dim not in valid_dimensions:
      raise ValueError('Unsupported parameter \'%s\'' % (dim))

  array_shape = tuple(parameters.values())
  dimensions = tuple(parameters.keys())
  coords = {}
  for dim in dimensions:
    # can change to something more sophisticated later
    if np.any(coordinates(dim, parameters[dim])):
      coords[dim] = coordinates(dim, parameters[dim])
  da = xr.DataArray(np.zeros(array_shape), dims=dimensions, coords=coords)

  for c in da.coords:
    da.coords[c].attrs['units'] = coord_units(c)
  return da


def new_model(num_samples, num_locations, num_time_steps,
              num_static_covariates, num_dynamic_covariates=0):
  """Return a xr.Dataset with an infection time series and a list of covariates.

  Args:
    num_samples: int representing the number of unique samples to run for each
      location
    num_locations: int representing the number of locations to model epidemics
      for
    num_time_steps: int representing the maximum number of time steps the
      epidemic can have
    num_static_covariates: int representing the number of static covariates for
      each location
    num_dynamic_covariates: int representing the number of dynamic covariates
      for each location (defaults to 0).

  Returns:
    ds: an xr.Dataset reprsenting the new infections and
      covariates for each location.
  """
  new_infections = new_dataarray({
      'sample': num_samples,
      'location': num_locations,
      'time': num_time_steps,
  })

  static_covariates = new_dataarray({
      'location': num_locations,
      'static_covariate': num_static_covariates
    })

  dynamic_covariates = new_dataarray({
      'location': num_locations,
      'time': num_time_steps,
      'dynamic_covariate': num_dynamic_covariates
    })

  ds = xr.Dataset({
      'new_infections': new_infections,
      'static_covariates': static_covariates,
      'dynamic_covariates': dynamic_covariates,
      })

  ds['new_infections'].attrs[
      'description'] = 'Number of new infections at each location over time.'
  ds['static_covariates'].attrs[
      'description'] = 'Static covariates at each location.'
  ds['dynamic_covariates'].attrs[
      'description'] = 'Dynamic covariates at each location and time.'
  return ds


def calculate_cumulative_infections(new_infections):
  """Calculate the cumulative infections over time.

  Args:
    new_infections: a xr.DataArray containing new_infections and dimension
      (samples, locations, time)

  Returns:
    cumulative_infections: a xr.DataArray containing the cumulative infections
    of dimension (samples, locations, time)
  """
  return new_infections.cumsum('time')


def calculate_total_infections(new_infections):
  """Calculate the total infections at each location and sample.

  Args:
    new_infections: a xr.DataArray containing new_infections and dimension
      (samples, locations, time)

  Returns:
    total_infections: a xr.DataArray containing the summed infections of
    dimension (samples, locations)
  """

  return new_infections.sum('time')


def calculate_infection_sums(ds):
  """Calculate the cumulative infections and total infections for a dataset.

  # TODO(shoyer) add clipping functions?

  Args:
    ds: a xr.Dataset that represents the trajectories
  Returns:
    summed_ds: a xr.Dataset with extra DataArrays added for the cumulative and
    total infections
  """
  ds['cumulative_infections'] = calculate_cumulative_infections(
      ds.new_infections)
  ds['cumulative_infections'].attrs['description'] = 'Cumulative infections'
  ds['total_infections'] = calculate_total_infections(ds.new_infections)
  ds['total_infections'].attrs['description'] = 'Total infections.'

  return ds
