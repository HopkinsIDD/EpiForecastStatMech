# Lint as: python3
"""Metrics for evaluating models."""

import xarray as xr


def train_test_split_time(data, split_day):
  """Split data into training_data (before split_day) and test_data (after).

  Args:
    data: an xr.Dataset containing the ground-truth infections.
    split_day: a time.coord at which to split the data into train and test sets.

  Returns:
    train_data: an xr.Dataset containing the training data for a model.
    test_data: an xr.Dataset containing the testing data for a model.
  """
  # everything before split_day
  train_data = data.sel(time=slice(None, split_day - 1))
  # everything after split_day
  test_data = data.sel(time=slice(split_day, None))
  # drop all variables that won't exist on real test data
  test_covariates = test_data[['location', 'time', 'static_covariate']]
  test_covariates['static_covariates'] = data['static_covariates']

  split_spec = ('train_test_split_time', dict(split_day=split_day))
  train_data.attrs['split'] = 'train'
  train_data.attrs['split_spec'] = split_spec
  test_covariates.attrs['split'] = 'test'
  test_covariates.attrs['split_spec'] = split_spec

  return train_data, test_covariates


def shard_locations_randomly(data, num_shards=5, seed=0):
  """Split data into roughly equally sized shards by location."""
  locations = data.location.values
  np.random.seed(seed)
  np.random.shuffle(locations)
  breakpoints = np.linspace(0, len(locations), num_shards, endpoint=False)
  breakpoints = breakpoints.astype(int)[1:]
  location_shards = np.split(locations, breakpoints)
  result = [data.sel(location=locs).copy() for locs in location_shards]
  for i, shard in enumerate(result):
    shard.attrs['shard'] = i
    shard.attrs['shard_spec'] = ('shard_locations_randomly',
                                 dict(num_shards=num_shards, seed=seed))
  return result
