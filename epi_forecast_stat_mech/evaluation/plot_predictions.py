# Lint as: python3
"""Plot helpers for predictions from high_level.Estimator.
"""

import itertools
from epi_forecast_stat_mech.evaluation.plot_constants import model_colors
from epi_forecast_stat_mech.evaluation.plot_constants import model_types
from matplotlib import pyplot as plt
import numpy as np


def plot_rollout_samples(predictions, model_to_plot, location_to_plot):
  """Helper function to plot the mean predicted value and all rollout samples.

  A helper function that plots the mean predicted number of new infections
  as a function of time, as well as plotting all of the different rollouts.

  Args:
    predictions: an xr.DataArray representing predicted new_infections with
      dimensions of (location, time, sample, model). Where model is the
       Estimator used to generate the predicted new_infections.
    model_to_plot: The coordinate name of the model that you want to plot.
    location_to_plot: The coordinate integer of the location that you want to
      plot.
  Returns:
    None
  """
  pred = predictions.sel(model=model_to_plot)
  mean = pred.isel(
      location=location_to_plot).mean('sample').rename('infection_mean')
  color_params = model_colors[model_to_plot]
  # is there a better way to do this?
  for i in pred.sample:
    plt.plot(
        pred.time,
        pred.isel({
            'location': location_to_plot,
            'sample': i
        }),
        alpha=0.1,
        label='_nolegend_',
        **color_params)
  plt.plot(pred.time, mean, markersize=2, label=model_to_plot, **color_params)
  return None


def plot_std_dev(predictions, model_to_plot, location_to_plot, num_stddevs=3):
  """Helper function to plot the mean predicted value and shade the error.

  A helper function that plots the mean predicted number of new infections
  as a function of time, as well as shades the area +/- num_stddevs around the
  mean.

  Args:
    predictions: an xr.DataArray representing predicted new_infections with
      dimensions of (location, time, sample, model). Where model is the
       Estimator used to generate the predicted new_infections.
    model_to_plot: The coordinate name of the model that you want to plot.
    location_to_plot: The coordinate integer of the location that you want to
      plot.
    num_stddevs: an int representing the number of standard deviations to shade.
      Defaults to 3.
  Returns:
    None
  """
  pred = predictions.sel({
      'model': model_to_plot
  }, drop=True).isel(
      location=location_to_plot, drop=True)
  mean = pred.mean('sample')
  stddev = pred.std('sample')
  upper = mean + num_stddevs * stddev
  lower = mean - num_stddevs * stddev
  color_params = model_colors[model_to_plot]
  plt.fill_between(pred.time.data, upper.data, lower.data, alpha=.2,
                   label='_nolegend_', color=color_params['color'])

  plt.plot(pred.time, mean, **color_params, label=model_to_plot)
  return None


def plot_observed_data(data_inf, predictions, location_to_plot):
  """Helper function to plot the observed data at a location.

  Args:
    data_inf: an xr.DataArray representing the *true* new_infections with
      dimensions of (location, time).
    predictions: an xr.DataArray representing predicted new_infections with
      dimensions of (location, time, sample, model). Where model is the
       Estimator used to generate the predicted new_infections.
    location_to_plot: The coordinate integer of the location that you want to
      plot.
  Returns:
    None
  """
  data_to_plot = data_inf.isel(location=location_to_plot)

  observed_color_params = model_colors['observed']
  plt.plot(
      data_to_plot.coords['time'].sel(
          time=(data_inf.time < min(predictions.time))),
      data_to_plot.sel(time=(data_inf.time < min(predictions.time))),
      **observed_color_params,
      label='observed')
  return None


def plot_ground_truth_data(data_inf, predictions, location_to_plot):
  """Helper function to plot the ground truth data at a location.

  Args:
    data_inf: an xr.DataArray representing the *true* new_infections with
      dimensions of (location, time).
    predictions: an xr.DataArray representing predicted new_infections with
      dimensions of (location, time, sample, model). Where model is the
       Estimator used to generate the predicted new_infections.
    location_to_plot: The coordinate integer of the location that you want to
      plot.
  Returns:
    None
  """
  data_to_plot = data_inf.isel(location=location_to_plot)
  ground_truth_color_params = model_colors['ground_truth']
  plt.plot(
      predictions.time,
      data_to_plot.sel(time=predictions.time),
      **ground_truth_color_params,
      label='ground truth')
  return None


def plot_one_model_predictions(data_inf,
                               predictions,
                               model_to_plot,
                               location_to_plot,
                               plot_pred_function=plot_rollout_samples,
                               plot_ground_truth=False):
  """Plot the data and predicted mean for a single model and location.

  Args:
    data_inf: an xr.DataArray representing the *true* new_infections with
      dimensions of (location, time).
    predictions: an xr.DataArray representing predicted new_infections with
      dimensions of (location, time, sample, model). Where model is the
       Estimator used to generate the predicted new_infections.
    model_to_plot: The coordinate name of the model that you want to plot.
    location_to_plot: The coordinate integer of the location that you want to
      plot.
    plot_pred_function: a function to plot the predicted mean values and some
      errors.
    plot_ground_truth: a boolean indicating whether to plot the ground truth
      new_infection values.
  Returns:
    None
  """
  plt.figure(figsize=(12, 8))
  plot_pred_function(predictions, model_to_plot, location_to_plot)
  plot_observed_data(data_inf, predictions, location_to_plot)
  if plot_ground_truth:
    plot_ground_truth_data(data_inf, predictions, location_to_plot)
  plt.xlabel('Time')
  plt.ylabel('New infections')
  plt.legend()
  plt.show()
  return None


def plot_many_model_predictions(data_inf,
                                predictions,
                                model_type_to_plot,
                                location_to_plot,
                                plot_pred_function=plot_rollout_samples,
                                plot_ground_truth=False):
  """Plot the data and predicted means for a type of model in one location.

  Args:
    data_inf: an xr.DataArray representing the *true* new_infections with
      dimensions of (location, time).
    predictions: an xr.DataArray representing predicted new_infections with
      dimensions of (location, time, sample, model). Where model is the
       Estimator used to generate the predicted new_infections.
    model_type_to_plot: The name of the model_type that you want to plot. Must
      be in plot_constants.model_type.keys().
    location_to_plot: The coordinate integer of the location that you want to
      plot.
    plot_pred_function: a function to plot the predicted mean values and some
      errors.
    plot_ground_truth: a boolean indicating whether to plot the ground truth
      new_infection values.
  Returns:
    None
  """
  plt.figure(figsize=(16, 8))
  for model_name in model_types[model_type_to_plot]:
    plot_pred_function(predictions, model_name, location_to_plot)
  plot_observed_data(data_inf, predictions, location_to_plot)
  if plot_ground_truth:
    plot_ground_truth_data(data_inf, predictions, location_to_plot)
  plt.xlabel('Time')
  plt.ylabel('New Infections')
  plt.legend()
  plt.show()


def final_size_comparison(data,
                          predictions_dict,
                          split_day,
                          figsize=(12, 8),
                          styles=('b.', 'r+', 'go')):
  styles = itertools.cycle(styles)
  predictions_time = None
  for predictions in predictions_dict.values():
    if predictions_time is None:
      predictions_time = predictions.time
    else:
      # We require all predictions in the dictionary to have aligned time.
      np.testing.assert_array_equal(predictions_time, predictions.time)

  final_size = data.new_infections.sel(time=predictions_time).sum('time')

  plt.figure(figsize=figsize)
  for style, (name, predictions) in zip(styles, predictions_dict.items()):
    K = predictions.mean('sample').sum('time')
    plt.plot(final_size, K, style, label=name)
  plt.plot(plt.xlim(), plt.xlim(), 'k--')
  plt.legend()
  plt.title('True vs. predicted final epidemic size (post day %d)' % split_day)
  plt.xlabel('True')
  plt.ylabel('Predicted')
  plt.show()
