"""A high-level interface for estimating sampling parameters for estimators."""

import copy
import functools
import logging

from epi_forecast_stat_mech import estimator_base
from epi_forecast_stat_mech import high_level
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability as tfp
import xarray as xr

tfp = tfp.experimental.substrates.jax
tfd = tfp.distributions


def _helper_sample_mech_params(rngkey, num_samples, initial_mech_params,
                               scale, rand_fun=tfd.Normal):
  """Helper function to randomly sample mechanistic parameters for rollouts.

  Takes the initial mech parameters, and randomly samples to generate
  num_samples of mechanistic parameters to use in the monte-carlo rollouts.

  Args:
   rngkey: A jax.random.PRNGKey to use to generate the parameters.
   num_samples: An int representing the number of rollout samples.
   initial_mech_params: A list of shape (num_locations, num_params)
     representing the mechanistic parameters for each location as determined
     by estimator.fit.
   scale: A sequence of floats of the same shape as initial_mech_params.
     Represents the scales passed to rand_fun for each mech_params.
   rand_fun: Optional. Function to use for generating random parameters.
  Returns:
    random_mech_params: A list of shape (num_location, num_samples,
      num_params) representing the mechanistic parameters to use for each
      location/rollout.
  """
  param_dist = rand_fun(loc=initial_mech_params, scale=jnp.exp(scale))
  params_sample_location_param = param_dist.sample(num_samples, seed=rngkey)
  params = jnp.swapaxes(params_sample_location_param, 1, 0)
  return params


def calc_total_infections(new_infections):
  """Calculate total infections to pass to the WIS function as validation.

  For computational simplicity we currently choose the observable to
  be the total number of infections at each location. We want to calibrate
  our scale parameters so that this quantitiy in our predictions `matches`
  the validation data. This is the quantity, where `matches` typically is
  quantile agreement (and can be changed by changing alpha.)

  Args:
    new_infections: An array of shape (location, time) that represents the new
      infections as a function of time.

  Returns:
    observed_data: An jnp.array of shape (location,) that represents
      the observed total infections at each location.
  """
  if isinstance(new_infections, xr.core.dataarray.DataArray):
    return jnp.asarray(new_infections.sum('time'))
  return new_infections.sum(axis=-1)  # bad if time is not last axis


def linear_minimize(f, center, direction, step_increments=None):
  if step_increments is None:
    step_increments = np.linspace(-5, 5, 11)
  values = np.array([f(center + i * direction) for i in step_increments])
  idx = np.argmin(values)
  i = step_increments[idx]
  return center + i * direction, values[idx], values


class AriadneEstimator(estimator_base.Estimator):
  """Fit parameters for other estimators to minimize the WIS."""

  def __init__(self,
               jesm_estimator,
               validation_time,
               train_steps,
               alpha=np.asarray([.02, .05, .1, .2, .3, .4, .5, .6, .7, .8,
                                 .9]),
               fused_train_steps=100,
               fit_seed=0,
               predict_seed=42,
               num_samples=10000,
               init_params=None,
               calculate_observable_fn=calc_total_infections,
               fit_all=True,):
    """An estimator that fits parameters for jesm_estimator to minimize the WIS.

    This is an estimator that is designed to produce reasonable(ish) error bars
    for any epi_forecast_stat_mech estimator. We accomplish this using the
    Weighted Interval Score (WIS) described here:
    https://arxiv.org/pdf/2005.12881.pdf

    Briefly, we consume an estimator with some fit mechanistic parametrs mp.
    Our goal is to find a range sigma_mp such that predictions sampled from
    this *range* of parameters minimizes the WIS.

    This estimator is named after Ariadne, a character in Inception who designs
    dreams within dreams.

    Args:
      jesm_estimator: an estimator_base.Estimator that you want to generate
        scale parameters for.
      validation_time: an int representing the length of time you want to
        predict.
      train_steps: an int representing the number of train_steps when we refit
        jesm_estimator.
      alpha: a jax.numpy.array representing the new PIs to use. Defaults to
        those used in FluSight.
      fused_train_steps: an int representing the number of train stesps to fuse
      fit_seed: an int used to seed the rng.
      predict_seed: an int used to seed the rng.
      num_samples: an int representing the number of samples to rollout at each
        location.
      init_params: an array of shape (mech_params,) representing the values to
        use as the initial guess for the optimizer.
      calculate_observable_fn: A function that accepts an array of shape
        (location, time) and returns an observable with shape (location,).
      fit_all: Bool. If true, refits all the estimators in place.
    """
    self.jesm_estimator = jesm_estimator
    self.validation_time = validation_time
    self.train_steps = train_steps
    self.alpha = alpha
    self.fused_train_steps = fused_train_steps
    self.fit_seed = fit_seed
    self.predict_seed = predict_seed
    self.num_samples = num_samples
    self.init_params = init_params
    self.calculate_observable_fn = calculate_observable_fn
    self.fit_all = fit_all

  def split_training_data(self, train_inf, validation_time):
    """Split the data into train and validation sets.

    We fit the scale parameters to minimize the WIS over the validation set.
    The validation set is the last validation_time observations from train_data.
    In the current setup, validation_time should equal predict_time, when we
    roll our predictions forwards.

    Args:
      train_inf: An xr.Dataset that represents some infection data and
        covariates.
      validation_time: An int that represents the number of time steps to use in
        our validation dataset.

    Returns:
      small_train_inf: An xr.Dataset that represents the new training data.
      validation_inf: An xr.Dataset that represents the validation data. The
        scale parameters are selected to minimize the WIS over this dataset.
    """
    total_time = len(train_inf.time)
    first_test_day_int = total_time - validation_time
    small_train_inf = train_inf.isel(time=slice(None, first_test_day_int - 1))
    validation_inf = train_inf.isel(time=slice(first_test_day_int, None))
    return small_train_inf, validation_inf

  def refit_estimator(self, train_data):
    """Refit our estimator on a smaller dataset.

    Args:
      train_data: an xr.Dataset representing the full training set
    """
    # Split train_data into small_train and validation
    logging.info('Refitting jesm_estimator on small dataset')

    small_train_data, validation_data = self.split_training_data(
        train_data, self.validation_time)
    self.train_data = train_data
    self.validation_data = validation_data

    # Retrain the estimator on just the small data
    # Is this necessary?
    self.small_estimator = copy.deepcopy(self.jesm_estimator)
    self.small_estimator.fit(small_train_data)
    self.refit_estimator_ = True
    return

  def _check_refitted(self):
    """Check if we have refit our estimator on the new, smaller train data."""
    if not hasattr(self, 'refit_estimator_') or not self.refit_estimator_:
      return False
    return True

  def update_validation_time(self, validation_time):
    """Update the split day.

    Args:
      validation_time: int representing the new split day
    """
    self.validation_time = validation_time
    self.refit_estimator_ = False
    self.is_trained_ = False
    return

  def update_alpha(self, alpha):
    """Update the alpha.

    Args:
      alpha: a jnp.array representing the new PIs to use
    """
    self.alpha = alpha
    self.is_trained_ = False
    return

  def update_num_samples(self, num_samples):
    """Update the number of samples to roll out.

    Args:
      num_samples: int representing the number of samples to roll out.
    """
    self.num_samples = num_samples
    self.is_trained_ = False
    return

  def _check_trained(self):
    if not hasattr(self, 'is_trained_') or not self.is_trained_:
      raise AttributeError('`fit` must be called before `predict`.')

  def fit(self, train_data):
    """Fit the scale for each mech_param to minimize the WIS.

    Args:
      train_data: An xr.Dataset representing the training data. We split this
        into train and validation data using self.validation_time.
    Returns:
      scale_params: A series of floats representing the scale of the Noraml
        distributions to sample mech_params from to minimize the WIS on the
        validation data.
    """

    def sample_based_weighted_interval_score(predicted_samples,
                                             validation,
                                             alpha,
                                             w=None):
      """Calculate the WIS using samples from a predictive distribution.

      Calculate the weighted interval score (WIS) using samples from a
      predictvie
      distribution. The original paper used a predictive distribution. Mcoram
      reimplmeneted the function to use *samples* from this distribution.

      Args:
        predicted_samples: An array of shape (location, samples) representing
          our predictions for the validation data. Must match the quantity of
          observed. If this is an xr.DataArray, one of the vmaps break.
        validation: An array of shape (location, ) representing our calculated,
          *true* quantity of interest at each location.
        alpha: A series of floats representing the prediction intervals to use.
          See https://arxiv.org/pdf/2005.12881.pdf Section 3.
        w: A series of floats representing the weights for each prediction
          interval. If None, defaults to alpha/2.

      Returns:
        wis: A float representing the weighted interval score.
      """
      half_alpha = alpha / 2.
      if w is None:
        w = half_alpha
      prob_cuts = jnp.concatenate((half_alpha, 1. - half_alpha))
      quantiles = jnp.quantile(predicted_samples, prob_cuts)
      lower, upper = jnp.split(quantiles, 2)
      interval_score = (upper - lower) + (
          jnp.where(validation < lower, lower - validation, 0.) +
          jnp.where(validation > upper, validation - upper, 0.)) / half_alpha
      wis = interval_score.dot(w) / len(interval_score)
      return wis

    def average_wis_of_observations(scale, rngkey, validation_inf):
      """Calculate the average WIS over all locations and samples.

      For a given set of validation_infections, calculate the WIS of our
      estimator
      predictions over num_samples.

      Args:
        scale: A series of floats of shape (num_mech_params, ) representing the
          scale of the Normal distribution to sample each mech_param from.
        rngkey: A jax.random.PRNGKey
        validation_inf: A xr.DataArray of shape (location, time) representing
          the infections in our validation window.

      Returns:
        average_wis: A float representing the WIS averaged over all samples and
          locations.
      """
      calculate_observable_fn = self.calculate_observable_fn
      scale = jnp.asarray([scale])
      observations = jnp.asarray(calculate_observable_fn(validation_inf))
      # Assumes estimator has been fit already
      self.small_estimator.sample_mech_params_fn = functools.partial(
          _helper_sample_mech_params,
          initial_mech_params=self.small_estimator.mech_params_for_jax_code,
          scale=scale,
          rand_fun=tfd.Normal)
      prediction_dist = self.small_estimator.predict(validation_inf,
                                                     self.num_samples)
      predictions = calculate_observable_fn(prediction_dist)
      result = jax.numpy.mean(
          jax.vmap(sample_based_weighted_interval_score,
                   in_axes=(0, 0, None))(predictions, observations, self.alpha))
      print(f'called average_wis with scale {scale}, got value {result}')
      return result

    # Refit the estimator if needed
    if not self._check_refitted() or self.fit_all:
      print('Something changed - refitting')
      self.refit_estimator(train_data)

    # Check that training data hasn't changed
    if not self.train_data.equals(train_data):
      print('Training data changed - refitting')
      self.refit_estimator(train_data)

    # Initialize the scales, they are the same for all locations
    init_params = self.init_params
    num_param = self.small_estimator.mech_params.shape[-1]
    if init_params is None:
      init_params = -1.0 * jnp.ones(num_param)

    key = jax.random.PRNGKey(self.fit_seed)
    key, subkey = jax.random.split(key)
    average_wis_partial = functools.partial(
        average_wis_of_observations,
        validation_inf=self.validation_data.new_infections,
        rngkey=subkey)

    criterion = average_wis_partial

    # First search diagonally.
    x = init_params
    direction = 0.5 * np.ones_like(x)
    x, _, _ = linear_minimize(criterion, x, direction)

    # Iteratively search along each dimension with progressively smaller
    # step sizes.
    num_iterations = 2
    directions = 0.2 * np.eye(len(x))
    for _ in range(num_iterations):
      for direction in directions:
        x, _, _ = linear_minimize(criterion, x, direction)
      directions /= 2.0

    self.is_trained_ = True
    # Return deltas
    self.scale_params = x

    try:
      self.jesm_estimator._check_fitted()
      fitted = True
    except AttributeError:
      fitted = False
    if not fitted or self.fit_all:
      print('fitting jesm estimator on all training data')
      self.jesm_estimator.fit(self.train_data)

    return self

  def predict(self, test_data, num_samples, seed=0):
    """Predict using new scale parameters."""
    self._check_trained()
    self.sample_mech_params_fn = functools.partial(
        _helper_sample_mech_params,
        initial_mech_params=self.jesm_estimator.mech_params_for_jax_code,
        scale=self.scale_params,
        rand_fun=tfd.Normal)
    self.jesm_estimator.sample_mech_params_fn = self.sample_mech_params_fn
    predictions = self.jesm_estimator.predict(
        test_data, num_samples=num_samples, seed=seed)
    return predictions


def get_estimator_dict(
    validation_time=48,
    train_steps=10000,
    num_samples=100,
    fit_all=True):
  estimator_dict = {}
  small_estimators = high_level.get_simple_estimator_dict()
  for name, small_estimator in small_estimators.items():
    if hasattr(small_estimator, 'mech_model'):
      estimator_dict['AE_' + name] = AriadneEstimator(
          jesm_estimator=small_estimator,
          validation_time=validation_time,
          train_steps=train_steps,
          num_samples=num_samples,
          fit_all=fit_all)
  return estimator_dict
