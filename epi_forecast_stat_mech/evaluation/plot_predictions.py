# Lint as: python3
"""Plot helpers for predictions from high_level.Estimator.
"""

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
  pred = predictions.sel(model=model_to_plot).dropna('time')
  mean = pred.isel(
      location=location_to_plot).mean('sample').rename('infection_mean')
  color_params = model_colors(model_to_plot)
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
      location=location_to_plot, drop=True).dropna('time')
  mean = pred.mean('sample')
  stddev = pred.std('sample')
  upper = mean + num_stddevs * stddev
  lower = mean - num_stddevs * stddev
  color_params = model_colors(model_to_plot)
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

  observed_color_params = model_colors('observed')
  max_observed_time = min(predictions.dropna('time').time)
  plt.plot(
      data_to_plot.coords['time'].sel(time=(data_inf.time < max_observed_time)),
      data_to_plot.sel(time=(data_inf.time < max_observed_time)),
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
  ground_truth_color_params = model_colors('ground_truth')

  plt.plot(
      predictions.dropna('time').time,
      data_to_plot.sel(time=predictions.dropna('time').time),
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


def plot_violin(ax, error_array, models_to_plot):
  """Make a violin plot of one error metric of multiple models on one dataset.

  Args:
    ax: A pyplot.axis object to draw on.
    error_array: An xr.DataArray representing the calculated errors of a given
      metric with dimensions of (location, time, model, value_type).
    models_to_plot: A list of strings representing the names of the models to
      plot. Must be elements of error_array.model.values().

  Returns:
    None
  """
  e = []

  for model in models_to_plot:
    mean_error = error_array.sel(model=model).mean('sample')
    mean_diff = mean_error.sel(value_type='difference')

  e.append(mean_diff)

  ax.violinplot(e, showextrema=True, showmeans=False)

  ax.set_ylabel('error, in raw counts', labelpad=None)

  ax.axhline(0, c='k')
  ax.set_xticks(np.arange(1, len(models_to_plot) + 1))
  ax.set_xticklabels(models_to_plot, rotation=15)
  ax.set_xlim(0.25, len(models_to_plot) + 0.75)
  ax.set_xlabel('Model')


def plot_scatter(ax, error_array, models_to_plot):
  """Make a scatter plot of real/pred metric of multiple models on one dataset.

  Args:
    ax: A pyplot.axis object to draw on.
    error_array: An xr.DataArray representing the calculated errors of a given
      metric, with dimensions of (location, time, model, value_type).
    models_to_plot: A list of strings representing the names of the models to
      plot. Must be elements of error_array.model.values().
  Returns:
    None
  """
  for model in models_to_plot:
    mean_error = error_array.sel(model=model).mean('sample')
    ax.scatter(mean_error.sel(value_type='ground_truth'),
               mean_error.sel(value_type='predicted'),
               label=model, s=2, alpha=0.75, c=model_colors(model)['color'])

  ax.set_aspect('equal')
  ax.set_ylim(ax.get_xlim())
  ax.set_ylabel('Predicted Value', labelpad=None)
  ax.set_xlabel('True Value', labelpad=None)
  ax.plot(ax.get_xlim(), ax.get_xlim(), 'k--')
  ax.legend()


def plot_violin_scatter(error_array, models_to_plot, metrics_to_plot):
  """Make a violin and scatter plot of metrics of multiple models on a dataset.

  Args:
    error_array: An xr.DataArray representing the calculated errors with
      with dimensions of (location, time, model, metric, value_type).
    models_to_plot: A list of strings representing the names of the models to
      plot. Must be elements of error_array.model.values().
    metrics_to_plot: A list of strings representing the names of the error
      metrics to plot. Must be elements of error_array.metric.values().
  Returns:
    None
  """

  fig, ax = plt.subplots(2, len(metrics_to_plot),
                         figsize=(5*len(metrics_to_plot), 15))

  for i, metric in enumerate(metrics_to_plot):
    plot_violin(ax[0][i], error_array.sel(metric=metric),
                models_to_plot)
    ax[0][i].set_title(metric)
    plot_scatter(ax[1][i], error_array.sel(metric=metric), models_to_plot)
    ax[1][i].set_title(metric)
  plt.show()
