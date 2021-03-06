# Lint as: python3
"""Metrics for evaluating models on simulated data."""

from absl import logging
import numpy as np
import xarray as xr


def _helper_data_validator(data_inf, predictions):
  # TODO(edklein, mcoram) decide if there's a better way to handle missing data
  # TODO(edklein, mcoram) more validation checks?
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


def _helper_construct_dataarray(ground_truth, predictions):
  """Helper function to construct an xr.DataArray with an 'value_type' dim.

  Args:
    ground_truth: an xr.DataArray containing the ground truth values
      for a give metric. Has dims (model, location, sample).
    predictions: an xr.DataArray containing the predicted values for a given
      metric. Has dims (model, location, sample).
  Returns:
    all_values: an xr.DataArray containing the ground truth and predicted values
      as well as their difference. Has dims (value_type, model, location,
      sample)
  """
  all_values = xr.concat(
      [ground_truth, predictions, ground_truth-predictions], dim='value_type')
  all_values.coords['value_type'] = ['ground_truth', 'predicted', 'difference']
  return all_values


def total_size_error(data_inf, predictions):
  """Calculate the error in the predicted cumulative epidemic size.

  Returns the error in the predicted vs. true cumulative epidemic sizes.
  The (signed) error is true - predicted.

  Args:
    data_inf: an xr.DataArray containing the ground-truth infections
    predictions: an xr.DataArray representing predicted new_infections with
      dimensions of (model, location, sample, time). Where model is the
       Estimator used to generate the predicted new_infections.
  Returns:
    total_size: an xr.DataArray containing the ground truth and predicted total
      sizes as well as their difference. Has dims (value_type, model, location,
      sample)
  """
  obs_plus_pred = _helper_data_validator(data_inf, predictions)

  true_total_size = data_inf.sum('time')

  # Calculate the predicted size
  pred_total_size = obs_plus_pred.sum('time')

  return _helper_construct_dataarray(true_total_size, pred_total_size)


def helper_time_percent_complete(inf, percent_complete=0.25):
  """Helper function to calculate when the infection is percent_complete.

  Args:
    inf: an xr.DataArray representing new_infections with
      dimensions of (location, sample, time).
    percent_complete: a float representing the percent of the epidemic that we
      want to see completed.
  Returns:
    true_time: an xr.DataArray containing the ground truth time of
      the percent complete. Has dims (location, sample).
  """
  # Calculate the total epidemic size
  total_size = inf.sum('time')
  target_size = percent_complete * total_size
  cumulative_infections = inf.cumsum('time')
  target_time = cumulative_infections.where(
      cumulative_infections <= target_size).argmax('time')
  return target_time


def time_percent_complete_error(data_inf, predictions, percent_complete=0.25):
  """Calculate the error in the predicted cumulative epidemic size.

  Returns the error in the predicted vs. true cumulative epidemic sizes.
  The (signed) error is true - predicted.

  Args:
    data_inf: an xr.DataArray containing the ground-truth infections
    predictions: an xr.DataArray representing predicted new_infections with
      dimensions of (model, location, sample, time). Where model is the
      Estimator used to generate the predicted new_infections.
    percent_complete: a float representing the percent of the epidemic that we
      want to see completed.
  Returns:
    time_percent_complete: an xr.DataArray containing the ground truth and
      predicted times as well as their difference. Has dims
      (value_type, model, location, sample)
  """
  obs_plus_pred = _helper_data_validator(data_inf, predictions)

  target_time = helper_time_percent_complete(obs_plus_pred, percent_complete)

  true_time = helper_time_percent_complete(data_inf, percent_complete)

  return _helper_construct_dataarray(true_time, target_time)


def cumulative_inf_error(data_inf, predictions, days_to_compare=14):
  """Calculate the error in the predicted cumulative infections for a time.

  Calculates the true and predicted cumulative infection rate from split_day
  to split_day+days_to_compare. Returns the error in the predicted vs. true
  infections. The (signed) error is true - predicted.

  Args:
    data_inf: an xr.DataArray containing the ground-truth infections
    predictions: an xr.DataArray representing predicted new_infections with
      dimensions of (model, location, sample, time). Where model is the
      Estimator used to generate the predicted new_infections.
    days_to_compare: an int representing the number of days (in simulation time)
      that we want to compare.
  Returns:
    cumulative_inf: an xr.DataArray containing the ground truth and
    predicted infection sizes as well as their difference. Has dims
    (value_type, model, location, sample)
  """
  if days_to_compare > len(predictions.time):
    raise ValueError('Days_to_compare is greater than the predicted times')
  small_predictions = predictions.isel(time=slice(None, days_to_compare))
  small_data = data_inf.sel(time=small_predictions.time)

  true_inf_size = small_data.sum('time')
  pred_inf_size = small_predictions.sum('time')

  return _helper_construct_dataarray(true_inf_size, pred_inf_size)


def peak_size_error(data_inf, predictions):
  """Calculate the error in the predicted maximum number of infections.

  Returns the error in the predicted vs. true peak epidemic sizes.
  Peak epidemic size is defined as the maximum number of new infections
  in a single time step.
  The (signed) error is true - predicted.

  Args:
    data_inf: an xr.DataArray containing the ground-truth infections
    predictions: an xr.DataArray representing predicted new_infections with
      dimensions of (model, location, sample, time). Where model is the
       Estimator used to generate the predicted new_infections.
  Returns:
    peak_size: an xr.DataArray containing the ground truth and predicted peak
      sizes as well as their difference. Has dims (value_type, model, location,
      sample)
  """
  true_peak_size = data_inf.max('time')

  # Sometimes we'll have already observed the infection peak
  # Need to account for this, so we'll look at the max between
  # observed and predicted
  obs_plus_pred = _helper_data_validator(data_inf, predictions)
  pred_peak_size = obs_plus_pred.max('time')

  return _helper_construct_dataarray(true_peak_size, pred_peak_size)


def peak_time_error(data_inf, predictions):
  """Calculate the error in the predicted maximum number of infections.

  Returns the error in the time of the predicted vs. true peak epidemic.
  Peak epidemic time is defined as the time coordinate with the maximum number
  of new infections in a single time step.
  The (signed) error is true - predicted.

  Args:
    data_inf: an xr.DataArray containing the ground-truth infections
    predictions: an xr.DataArray representing predicted new_infections with
      dimensions of (location, time, sample, model). Where model is the
       Estimator used to generate the predicted new_infections.
  Returns:
    peak_time: an xr.DataArray containing the ground truth and predicted peak
      times as well as their difference. Has dims (value_type, model, location,
      sample)
  """
  true_peak_time = data_inf.argmax('time', skipna=True)

  obs_plus_pred = _helper_data_validator(data_inf, predictions)
  pred_peak_time = obs_plus_pred.argmax('time', skipna=True)

  return _helper_construct_dataarray(true_peak_time, pred_peak_time)
