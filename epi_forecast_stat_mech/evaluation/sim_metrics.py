# Lint as: python3
"""Metrics for evaluating models on simulated data."""

import xarray as xr


def epidemic_size_error(data_inf, pred_inf):
  """Calculate the error in the predicted cumulative epidemic size.

  Returns the error in the predicted vs. true cumulative epidemic sizes.
  The (signed) error is predicted - true.

  Args:
    data_inf: an xr.DataArray containing the ground-truth infections
    pred_inf: an xr.DataArray containing the predicted infections from
      a model
  Returns:
    pred_errors: an xr.DataArray containing errors in the predicted cumulative
      infections.
  """
  true_epidemic_size = data_inf.sum('time')
  # predictions don't necessarily include already observed infections
  # so trim the data to match the predicted times.
  # This doesn't change the errors, as the observed infections cancel
  eval_inf = data_inf.sel(time=pred_inf.time)
  true_epidemic_size = eval_inf.sum('time')
  pred_epidemic_size = pred_inf.sum('time')
  pred_errors = pred_epidemic_size - true_epidemic_size
  return pred_errors


def peak_size_error(data_inf, pred_inf):
  """Calculate the error in the predicted maximum number of infections.

  Returns the error in the predicted vs. true peak epidemic sizes.
  Peak epidemic size is defined as the maximum number of new infections
  in a single time step.
  The (signed) error is predicted - true.

  Args:
    data_inf: an xr.DataArray containing the ground-truth infections
    pred_inf: an xr.DataArray containing the predicted infections from
      a model
  Returns:
    true_error: an xr.DataArray containing the error in the predicted
      peak size, or 0 if the peak was already observed.
  """
  true_peak_size = data_inf.max('time')
  pred_peak_size = pred_inf.max('time')

  # Sometimes we'll have already observed the infection peak
  # Need to account for this, so we'll look at the max between
  # observed and infected.
  eval_inf = data_inf.sel(time=pred_inf.time)
  eval_peak_size = eval_inf.max('time')
  pred_error = pred_peak_size - true_peak_size
  true_error = pred_error.where(eval_peak_size >= true_peak_size, 0)

  return true_error
