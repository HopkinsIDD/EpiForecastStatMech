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


def train_model(data, estimator, split_function, split_args, train_steps=10000,
                num_samples=1):
  """Train a model.

  Args:
    data: an xr.Dataset containing the ground-truth infections
    estimator: a high_level.Estimator to train and predict infection curves
    split_function: a data_rollouts.split function that splits data into train
      and test sets.
    split_args: a tuple of arguments to pass to split_function, except data.
    train_steps: an int specivying the number of training steps.
    num_samples: an int specifying the number of mc samples to run.
  Returns:
    predictions: an xr.Dataset containing all the predictions of estimator on
      test data.
  """
  train_data, test_data = split_function(data, *split_args)
  trained_estimator = estimator.fit(train_data, train_steps)
  predictions = trained_estimator.predict(
      len(test_data.time), num_samples=num_samples)
  return predictions


def evaluate_model(predictions, metrics):
  """Evaluate a model's predictions using some evaluation metrics.

  Args:
    predictions: an xr.Dataset containing the predicted infections
    metrics: a dictionary where the key is the name of the metric, and the
      value is a partial object that evaluates the metric.
  Returns:
    eval_data: an xr.Dataset containing all the predictions and all metrics as
      DataArrays.
  """
  eval_data = predictions.copy()
  for metric_name in metrics:
    eval_data[metric_name] = metrics[metric_name](predictions.new_infections)
  return eval_data


def train_and_evaluate_model(data, estimator, split_function, split_args,
                             train_steps, num_samples, metrics):
  """Train a model and evaluate it using some evaluation metrics.

  Args:
    data: an xr.Dataset containing the ground-truth infections
    estimator: a high_level.Estimator to train and predict infection curves
    split_function: a data_rollouts.split function that splits data into train
      and test sets.
    split_args: a tuple of arguments to pass to split_function, except data
    train_steps: an int specifying the number of training steps
    num_samples: an int specifying the number of mc samples to run.
    metrics: a dictionary where the key is the name of the metric, and the
      value is a partial object that evaluates the metric.
  Returns:
    eval_data: an xr.Dataset containing all the predictions and all metrics as
      DataArrays.
  """
  predictions = train_model(data, estimator, split_function, split_args,
                            train_steps, num_samples)
  # TODO(edklein) add decorator that runs averages the samples then runs
  # single-sample evaluation metic
  eval_data = evaluate_model(predictions, metrics)
  return eval_data


def evaluate_model_sequentially(data, estimator, split_function, split_args,
                                train_steps, num_samples, metrics):
  """Train the model on sequentially more data.

  Args:
    data: an xr.Dataset containing the ground-truth infections
    estimator: a high_level.Estimator to train and predict infection curves
    split_function: a function that splits data into train and test sets.
    split_args: an xr.DataArray of arguments to pass to split_function, except
      data, indicating *all* values to split the Data along.
    train_steps: an int specifying the number of training steps
    num_samples: an int specifying the number of mc samples to run.
    metrics: a dictionary where the key is the name of the metric, and the
      value is a partial object that evaluates the metric.
  Returns:
    eval_data: a list of xr.Datasets containing all the predictions and all
      metrics as Datasets.
  """

  # TODO(edklein) vectorize if possible
  # TODO(edklein) if split_function is None treat entire dataset as training

  # @hack
  # assume split_args is 1d
  if len(split_args.dims) != 1:
    raise ValueError('Split_arguments must have only one dimension')
  dim = split_args.dims[0]
  eval_data = xr.Dataset(coords=data.coords)
  for split_arg in split_args:
    name = str(split_arg.coords[dim].data)
    eval_dataset = train_and_evaluate_model(data, estimator, split_function,
                                            (split_arg.data,), train_steps,
                                            num_samples, metrics)
    # @ half a hack
    var_names = {}
    for var in eval_dataset.data_vars:
      var_names[var] = str(var)+'_'+str(name)
    # TODO(edklein): change this so info gets added along a new axis
    eval_data = eval_data.merge(eval_dataset.rename(var_names))

  return eval_data

