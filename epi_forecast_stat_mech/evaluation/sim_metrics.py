# Lint as: python3
"""Metrics for evaluating models on simulated data."""

from absl import logging
import numpy as np
import xarray as xr


def _helper_data_validator(data_inf, predictions):
  # TODO(eklein, mcoram) decide if there's a better way to handle missing data
  """Function to make sure that data and predictions cover the same time.

  Args:
    data_inf: an xr.DataArray containing the ground-truth infections
    predictions: an xr.DataArray representing predicted new_infections with
      dimensions of (location, time, sample, model). Where model is the
       Estimator used to generate the predicted new_infections.
  Returns:
    obs_plus_pred: an xr.Daraset containing the observed and predicted new
      infections at each point in time.
  Raises:
    ValueError: if predictions does not have results for as far into the future
      as data does.
  """
  # TODO(edklein) proper handling of null and/or nan values
  data_end_time = data_inf.time.max()
  pred_end_time = predictions.time.max()

  if data_end_time > pred_end_time:
    raise ValueError('Data extend farther into the future than predictions')

  if np.any(data_inf.time != predictions.time):
    logging.warning('The times in data do not match the times in predicted')

  observed_data = data_inf.sel(
      time=(data_inf.time < predictions.time.min()))
  obs_plus_pred = xr.concat(
      xr.broadcast(observed_data, predictions, exclude=['time']), dim='time')

  obs_plus_pred_clip = obs_plus_pred.sel(
      time=(obs_plus_pred.time <= data_inf.time.max()))

  return obs_plus_pred_clip


def total_size_error(data_inf, predictions):
  """Calculate the error in the predicted cumulative epidemic size.

  Returns the error in the predicted vs. true cumulative epidemic sizes.
  The (signed) error is predicted - true.

  Args:
    data_inf: an xr.DataArray containing the ground-truth infections
    predictions: an xr.DataArray representing predicted new_infections with
      dimensions of (location, time, sample, model). Where model is the
       Estimator used to generate the predicted new_infections.
  Returns:
    true_total_size: an xr.DataArray containing the ground truth total
      epidemic size. Has dims (location, sample, model).
    pred_total_size: an xr.DataArray containing the predicted total
      epidemic size. Has dims (location, sample, model).
    error_total_size: an xr.DataArray containing the error in the predicted
      total epidemic size.
  """
  obs_plus_pred = _helper_data_validator(data_inf, predictions)

  true_total_size = data_inf.sum('time')

  # Calculate the predicted size
  pred_total_size = obs_plus_pred.sum('time')

  return true_total_size, pred_total_size, pred_total_size - true_total_size


def helper_time_percent_complete(inf, percent_complete=0.25):
  """Helper function to calculate when the infection is percent_complete.

  Args:
    inf: an xr.DataArray representing new_infections with
      dimensions of (location, time, sample).
    percent_complete: a float representing the percent of the epidemic that we
      want to see completed.
  Returns:
    true_time: an xr.DataArray containing the ground truth time of
      the percent complete. Has dims (location, sample).
  """
  # Calculate the total epidemic size
  total_size = inf.sum('time')
  target_size = percent_complete * total_size
  cumulative_infections = inf.cumsum()
  target_time = cumulative_infections.where(
      cumulative_infections <= target_size).argmax('time')
  return target_time


def time_percent_complete_error(data_inf, predictions, percent_complete=0.25):
  """Calculate the error in the predicted cumulative epidemic size.

  Returns the error in the predicted vs. true cumulative epidemic sizes.
  The (signed) error is predicted - true.

  Args:
    data_inf: an xr.DataArray containing the ground-truth infections
    predictions: an xr.DataArray representing predicted new_infections with
      dimensions of (location, time, sample, model). Where model is the
      Estimator used to generate the predicted new_infections.
    percent_complete: a float representing the percent of the epidemic that we
      want to see completed.
  Returns:
    true_time: an xr.DataArray containing the ground truth time of
      the percent complete. Has dims (location, sample, model).
    target_time: an xr.DataArray containing the predicted time of the
      percent complete. Has dims (location, sample, model).
    error: an xr.DataArray containing the error in the predicted
      time of the percent complete. Has dims (location, sample, model).
  """
  obs_plus_pred = _helper_data_validator(data_inf, predictions)

  target_time = helper_time_percent_complete(obs_plus_pred, percent_complete)

  true_time = helper_time_percent_complete(data_inf, percent_complete)

  return true_time, target_time, target_time - true_time


def peak_size_error(data_inf, predictions):
  """Calculate the error in the predicted maximum number of infections.

  Returns the error in the predicted vs. true peak epidemic sizes.
  Peak epidemic size is defined as the maximum number of new infections
  in a single time step.
  The (signed) error is predicted - true.

  Args:
    data_inf: an xr.DataArray containing the ground-truth infections
    predictions: an xr.DataArray representing predicted new_infections with
      dimensions of (location, time, sample, model). Where model is the
       Estimator used to generate the predicted new_infections.
  Returns:
    true_peak_size: an xr.DataArray containing the ground truth peak size of the
      epidemic. Has dims (location, sample, model).
    pred_peak_size: an xr.DataArray containing the predicted peak size of the
      epidemic. Has dims (location, sample, model).
    error_peak_size: an xr.DataArray containing the error in the predicted
      peak epidemic size. Has dims (location, sample, model).
  """
  true_peak_size = data_inf.max('time')

  # Sometimes we'll have already observed the infection peak
  # Need to account for this, so we'll look at the max between
  # observed and predicted
  obs_plus_pred = _helper_data_validator(data_inf, predictions)
  pred_peak_size = obs_plus_pred.max('time')

  return true_peak_size, pred_peak_size, pred_peak_size-true_peak_size


def peak_time_error(data_inf, predictions):
  """Calculate the error in the predicted maximum number of infections.

  Returns the error in the time of the predicted vs. true peak epidemic.
  Peak epidemic time is defined as the time coordinate with the maximum number
  of new infections in a single time step.
  The (signed) error is predicted - true.

  Args:
    data_inf: an xr.DataArray containing the ground-truth infections
    predictions: an xr.DataArray representing predicted new_infections with
      dimensions of (location, time, sample, model). Where model is the
       Estimator used to generate the predicted new_infections.
  Returns:
    true_peak_time: an xr.DataArray containing the ground truth peak time of the
      epidemic. Has dims (location, sample, model).
    pred_peak_time: an xr.DataArray containing the predicted peak time of the
      epidemic. Has dims (location, sample, model).
    error: an xr.DataArray containing the error in the predicted
      peak time. Has dims (location, sample, model).
  """
  true_peak_time = data_inf.argmax('time', skipna=True)

  obs_plus_pred = _helper_data_validator(data_inf, predictions)
  pred_peak_time = obs_plus_pred.argmax('time', skipna=True)

  return true_peak_time, pred_peak_time, pred_peak_time - true_peak_time
