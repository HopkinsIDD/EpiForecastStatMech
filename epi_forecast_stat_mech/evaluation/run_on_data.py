# Lint as: python3
"""Metrics for evaluating models."""

import xarray as xr


def train_test_split_time(data, split_day):
  """Split data into training_data (before split_day) and test_data (after).

  Args:
    data: an xr.Dataset containing the ground-truth infections.
    split_day: a time.coord at which to split the data into
      train and test sets.
  Returns:
    train_data: an xr.Dataset containing the training data for a model.
    test_data: an xr.Dataset containing the testing data for a model.
  """
  # everything before split_day
  train_data = data.sel(time=slice(None, split_day-1))
   # everything after split_day
  test_data = data.sel(time=slice(split_day, None))
  # drop all variables that won't exist on real test data
  test_covariates = test_data[['location', 'time', 'static_covariate']]
  test_covariates['static_covariates'] = data['static_covariates']
  return train_data, test_covariates
