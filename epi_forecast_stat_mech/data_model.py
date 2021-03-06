# Lint as: python3
"""Data model for representing real data or simulated data.

Define an xarray.DataSeries to hold new_infection trajectories over time
and associated covariates. This is our data model that will be used for
'real' and simulated data. Define helper functions to calculate useful
secondary quantities and store them in the DataSeries.
"""
from .constants import coord_units
from .constants import coordinates
from absl import logging
import numpy as np
import re
import xarray as xr

valid_dimensions = [
    'location', 'time', 'static_covariate', 'dynamic_covariate', 'model'
]

TIME_ZERO = np.datetime64('2019-12-31')


def new_dataarray(parameters):
  """Helper function to create xr.DataArrays.

  Create xr.DataArrays with consistent dimensions, coordinates, and units.

  Args:
    parameters: a dictionary of (dimension: number) pairs. Dimension must be in
      valid_dimensions. Number is the length of that axis.

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


def new_model(num_locations,
              num_time_steps,
              num_static_covariates,
              num_dynamic_covariates=0):
  """Return a xr.Dataset with an infection time series and a list of covariates.

  Args:
    num_locations: int representing the number of locations to model epidemics
      for
    num_time_steps: int representing the maximum number of time steps the
      epidemic can have
    num_static_covariates: int representing the number of static covariates for
      each location
    num_dynamic_covariates: int representing the number of dynamic covariates
      for each location. If zero, this DataArray is not created.

  Returns:
    ds: an xr.Dataset reprsenting the new infections and
      covariates for each location.
  """
  new_infections = new_dataarray({
      'location': num_locations,
      'time': num_time_steps,
  })

  static_covariates = new_dataarray({
      'location': num_locations,
      'static_covariate': num_static_covariates
  })

  ds = xr.Dataset({
      'new_infections': new_infections,
      'static_covariates': static_covariates,
  })

  ds['new_infections'].attrs[
      'description'] = 'Number of new infections at each location over time.'
  ds['static_covariates'].attrs[
      'description'] = 'Static covariates at each location.'

  # Only create this array if we have dynamic_covariates
  # Otherwise we cannot save/load this dataset
  if num_dynamic_covariates > 0:
    dynamic_covariates = new_dataarray({
        'location': num_locations,
        'time': num_time_steps,
        'dynamic_covariate': num_dynamic_covariates
    })
    ds['dynamic_covariates'] = dynamic_covariates
    ds['dynamic_covariates'].attrs[
        'description'] = 'Dynamic covariates at each location and time.'

  return ds


def _helper_shift_dataarray(shifts, array_to_shift):
  """Helper function to shift array_to_shift by shift amount.

  Args:
    shifts: a np.array of shape (location, ) containing the values to shift by.
    array_to_shift: an xr.DataArray with dimensions (location,) to shift

  Returns:
    shifted_array: a copy of array_to_shift with the values shifted in time by
      shifts.
    shift_dataarray: an xr.DataArray of dimension (location,) containing
      the times we shifted by.
  """

  shift_dataarray = xr.DataArray(shifts, dims=['location'])
  old_ni = array_to_shift.copy()
  shifted_array = xr.concat([
      old_ni.isel(location=k).shift(time=shifts[k], fill_value=0)
      for k in range(old_ni.sizes['location'])
  ],
                            dim='location')

  # Count the number of trajectories we don't shift,
  # raise a warning if we exceed 1/4 of all trajectories
  num_not_shifted = xr.where(shift_dataarray == 0, 1, 0).sum(['location'])

  if num_not_shifted > (len(array_to_shift.location) / 4):
    logging.warning(
        'More than 1/4 of the trajectories were not shifted in time in %d '
        'locations or samples. Consider changing SPLIT_TIME.',
        (num_not_shifted.values))

  return shifted_array, shift_dataarray


def _first_true(arr, axis):
  return np.apply_along_axis(lambda x: np.where(x)[0][0], axis=axis, arr=arr)


def shift_timeseries_by(data, shifts_all, enforce_positive_shifts: bool = True):
  """Return a copy of data with shifted infection start times.

  Returns a copy of data with where time and location  dependent DataArrays are
  shifted by a location dependent shift.

  Args:
    data: a xr.Dataset of the simulated infections over time.
    shifts_all: The integer amount to shift time by in each location.
    enforce_positive_shifts: When true, treat negative shifts as shifts by 0.

  Returns:
    trajectories: a copy of data with the start times of new_infections shifted.
  """
  trajectories = data.copy()

  if enforce_positive_shifts:
    # We don't want to shift any infection curves so they start before time 0
    shifts = np.where(shifts_all > 0, shifts_all, 0)
  else:
    shifts = shifts_all

  for d_var in trajectories.data_vars:
    if 'time' in trajectories[d_var].dims:
      if not 'location' in trajectories[d_var].dims:
        raise ValueError('Only location dependent time-shifts are allowed; '
                         f'{d_var} doesn\'t depend on location')
      shifted_d_var, shift_dataarray = _helper_shift_dataarray(
          shifts, trajectories[d_var])
      trajectories[d_var] = shifted_d_var
      trajectories['start_time'] = shift_dataarray

  return trajectories


def shift_timeseries(data, fraction_infected_limits, split_time):
  """Return a copy of data with shifted infection start times.

  Returns a copy of data with the start times of new_infections shifted so that
  the percent complete of the infections is randomly distributed between
  fraction_infected_limits on split_time.

  Args:
    data: a xr.Dataset of the simulated infections over time.
    fraction_infected_limits: A pair of floats in [0, 1] representing the limits
      on the fraction of the population that will be infected at split_time.
    split_time: a value of data.time.values() representing the day at which we
      split the train and test data.

  Returns:
    trajectories: a copy of data with the start times of new_infections shifted.
  """
  trajectories = data.copy()

  num_locations = len(trajectories.location.values)
  # Randomly generate the fraction of infected people for each
  # location.
  if 'fraction_infected' in trajectories.data_vars:
    trajectories['fraction_infected'].data = np.random.uniform(
        fraction_infected_limits[0], fraction_infected_limits[1], num_locations)
  else:
    trajectories['fraction_infected'] = xr.DataArray(
        np.random.uniform(fraction_infected_limits[0],
                          fraction_infected_limits[1], num_locations),
        dims=['location'])

  cases = trajectories.new_infections.fillna(0).cumsum('time')
  trajectories['final_size'] = cases.isel(time=-1)
  target_cases = (trajectories['fraction_infected'] *
                  trajectories['final_size']).round()
  hit_times = (cases >= target_cases).reduce(_first_true, dim='time')
  shifts_all = split_time - hit_times

  return shift_timeseries_by(trajectories, shifts_all)


def validate_data(data,
                  enable_regex_check=False,
                  require_dynamics=False,
                  require_samples=False,
                  require_no_samples=False,
                  require_integer_time=True,
                  require_canonical_split_time=True):
  """Check the validity of the data xarray."""
  # TODO(mcoram): Consider validating data that has a model dim.
  # TODO(edklein): Consider removing require_samples.
  if require_samples and require_no_samples:
    raise ValueError('invalid call: only require one of samples or no_samples.')
  if not isinstance(data, xr.Dataset):
    raise ValueError('data must be an xarray')

  # Check for required data_vars.
  required_data_vars_set = set(['new_infections', 'static_covariates'])
  if require_dynamics:
    required_data_vars_set.add('dynamic_covariates')
  if require_canonical_split_time:
    required_data_vars_set.add('canonical_split_time')
  data_vars_set = set(data.data_vars.keys())
  missing_data_vars = required_data_vars_set.difference(data_vars_set)
  if missing_data_vars:
    helper_strs = []
    if 'canonical_split_time' in missing_data_vars:
      helper_strs.append('To add a sensible default canonical_split_time, run '
                         'data_model.set_canonical_split_time on your data.')
    raise ValueError('data is missing required data_vars: %s\n%s' %
                     (missing_data_vars, '\n'.join(helper_strs)))

  required_dims_set = set(['location', 'time', 'static_covariate'])
  if require_samples:
    required_dims_set = required_dims_set.union(set(['sample']))

  new_infection_dims = set(data.new_infections.dims)
  if require_no_samples:
    if new_infection_dims != set(('location', 'time')):
      raise ValueError(
          '`data.new_infections` is required to have dims `(\'location\', \'time\')`; '
          'got {data.new_infections.dims}.'.format(data=data))
  if require_samples:
    if new_infection_dims != set(('location', 'time', 'sample')):
      raise ValueError(
          '`data.new_infections` is required to have dims `(\'location\', \'time\', \'sample\')`; '
          'got {data.new_infections.dims}.'.format(data=data))
  if not (require_samples or require_no_samples):
    acceptable_dims = set(('location', 'time', 'sample'))
    other_dims = new_infection_dims.difference(acceptable_dims)
    if other_dims:
      raise ValueError(
          '`data.new_infections` is required to have dims `(\'location\', \'time\')  or `(\'location\', \'time\', \'sample\')`; '
          'got {data.new_infections.dims}.'.format(data=data))

  if set(data.static_covariates.dims) != set(('location', 'static_covariate')):
    raise ValueError(
        '`data.static_covariates` is required to have dims `(\'location\', \'static_covariate\')`; '
        'got {data.static_covariates.dims}.'.format(data=data))
  if 'dynamic_covariates' in data_vars_set:
    required_dims_set = required_dims_set.union(set(['dynamic_covariate']))
    if set(data.dynamic_covariates.dims) != set(
        ('location', 'time', 'dynamic_covariate')):
      raise ValueError(
          '`data.dynamic_covariates` is required to have dims `(\'location\', \'time\', \'dynamic_covariate\')`; '
          'got {data.dynamic_covariates.dims}.'.format(data=data))
  # Check for required dims.
  data_dims_set = set(data.dims.keys())
  missing_dims = required_dims_set.difference(data_dims_set)
  if missing_dims != set():
    raise ValueError('data is missing required dims: %s' % (missing_dims,))

  if require_integer_time and not np.issubdtype(data.time.dtype, np.integer):
    raise ValueError('`data.time` is required to be integer. Please run '
                     'data_model.convert_data_to_integer_time on this data.')

  # Make requirements about null patterns.
  static_covariates_null = data.static_covariates.transpose(
      'location', 'static_covariate').isnull()
  if static_covariates_null.sum():
    raise ValueError('null data is not allowed in data.static_covariates.')
  if require_dynamics:
    dynamic_covariates_null = data.dynamic_covariates.transpose(
        'location', 'time', 'dynamic_covariate').isnull()
    if dynamic_covariates_null.sum():
      raise ValueError('null data is not allowed in data.dynamic_covariates.')
  new_infections_null = data.new_infections.transpose('location', 'time',
                                                      ...).isnull()
  if enable_regex_check:
    acceptable_null_pattern = re.compile(r'^1*0*1*$')
    bad_locations_accum = []
    for location, time_null in new_infections_null.groupby('location'):
      # convert to a string of 0's and 1's.
      time_null_str = ''.join(time_null.astype('int32').astype('str').data)
      if not re.match(acceptable_null_pattern, time_null_str):
        bad_locations_accum.append(location)
    if bad_locations_accum:
      raise ValueError(
          'data.new_infections.isnull() has a bad pattern at locations: %s.' %
          (bad_locations_accum,))
  # Validate no new_infections are < 0.
  new_infections = data.new_infections.transpose('location', 'time', ...)
  bad_new_infections = new_infections < 0
  if bad_new_infections.sum().item() > 0:
    raise ValueError('data.new_infections contains negative entries.')


def validate_data_for_fit(data, **kwargs):
  validate_data(
      data,
      require_no_samples=True,
      require_canonical_split_time=False,
      **kwargs)


def calculate_cumulative_infections(new_infections):
  """Calculate the cumulative infections over time.

  Args:
    new_infections: a xr.DataArray containing new_infections and dimension
      (locations, time)

  Returns:
    cumulative_infections: a xr.DataArray containing the cumulative infections
    of dimension (locations, time)
  """
  return new_infections.cumsum('time')


def calculate_total_infections(new_infections):
  """Calculate the total infections at each location.

  Args:
    new_infections: a xr.DataArray containing new_infections and dimension
      (locations, time)

  Returns:
    total_infections: a xr.DataArray containing the summed infections of
    dimension (locations)
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


def _get_delta(series):
  """Returns the (assumed uniform) spacing between consecutive entries."""
  return np.diff(series.values[:2])[0]


def datetime_to_int(date, ds=None):
  """Convert datetimes to integer date indices."""
  if ds is None:
    return (date - TIME_ZERO) // np.timedelta64(1, 'D')
  if not hasattr(ds, 'original_time'):
    raise ValueError('Dataset with no original_time cannot be used to '
                     'convert datetimes to ints.')
  first_int = ds.time.values[0]
  delta_int = _get_delta(ds.time)
  first_date = ds.original_time.values[0]
  delta_date = _get_delta(ds.original_time)
  result = first_int + ((date - first_date) / delta_date) * delta_int
  if hasattr(result, 'dtype'):
    result = result.astype(int)
  else:
    result = int(result)
  return result


def int_to_datetime(int_date, ds=None):
  """Convert integer date indices to datetimes."""
  if ds is None:
    return TIME_ZERO + int_date * np.timedelta64(1, 'D')
  if not hasattr(ds, 'original_time'):
    raise ValueError('Dataset with no original_time cannot be used to '
                     'convert ints to datetimes.')
  first_int = ds.time.values[0]
  delta_int = _get_delta(ds.time)
  first_date = ds.original_time.values[0]
  delta_date = _get_delta(ds.original_time)
  return first_date + ((int_date - first_int) / delta_int) * delta_date


def compute_integer_time(trajectories):
  """Define integer time 0 as the time of the first infection."""
  time = trajectories.time
  if len(time) == 0:
    return time.astype('long')
  has_infections = trajectories.new_infections.sum('location', skipna=True) > 0
  if not has_infections.any():
    time0 = time.isel(time=-1)
  else:
    time0 = time.isel(time=np.where(has_infections.data)[0][0])
  if len(time) == 1:
    if np.issubdtype(time.dtype, np.datetime64):
      delta_t = np.timedelta64(1, 'D')
    else:
      delta_t = 1
  else:
    diff_time = time.diff('time')
    delta_t = diff_time[0].item()
    assert (diff_time == diff_time[0]).all().item()
  integer_time = ((time - time0).astype('long') // delta_t)
  return integer_time


def compute_numpy_index_time(trajectories):
  """numpy_index_time is just np.arange. Avoid this if possible."""
  raw_time = np.arange(trajectories.sizes['time'])
  numpy_time = xr.DataArray(
      raw_time, dims=('time',), coords=dict(time=raw_time))
  return numpy_time


def convert_data_to_integer_time(trajectories, method='days_in_2020'):
  """Store original_time and replace time with integer time."""
  if trajectories.coords.get('original_time', None):
    assert np.issubdtype(trajectories.time.dtype, np.integer), (
        'Integer time expected, because original_time is present.')
    return trajectories
  time = trajectories.time
  if method == 'days_in_2020':
    integer_time = datetime_to_int(time)
  elif method == 'numpy_index':
    integer_time = compute_numpy_index_time(trajectories)
  elif method == 'days_from_first_case':
    integer_time = compute_integer_time(trajectories)
  else:
    raise ValueError('method must be one of "days_in_2020", "numpy_index", or '
                     '"days_from_first_case".')
  trajectories = trajectories.copy()
  trajectories['original_time'] = time
  trajectories['time'] = integer_time
  if hasattr(trajectories, 'canonical_split_time'):
    orig_split_time = trajectories['canonical_split_time']
    trajectories['original_canonical_split_time'] = orig_split_time
    idx = np.searchsorted(trajectories['original_time'], orig_split_time)
    trajectories['canonical_split_time'] = trajectories['time'][idx]
  return trajectories


def set_sensible_canonical_split_time(data):
  peak_idx = data['new_infections'].argmax('time', skipna=True)
  peak_time = data.time[peak_idx].drop('time')
  peak_times = np.sort(peak_time.values.ravel())
  data['canonical_split_time'] = peak_times[len(peak_times) // 2]
