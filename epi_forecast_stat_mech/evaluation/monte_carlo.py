# Lint as: python3
"""Code for drawing Monte Carlo rollouts from models."""

import functools

import jax
from jax import numpy as jnp


def trajectories(rollout_fn, rng, args):
  """Computes batches of `nsamples` trajectories for all `args`.

  Args:
    rollout_fn: a callable that takes `(rng, args)` and returns an array.
    rng: a `jax.random.PRNGKey`.
    args: a pytree of `args` that can be passed to `rollout_fn`. All args will
      be vmapped along axis 0.

  Returns:
    An array of shape `[batch, nsamples] + s` where `batch` is the size of
    dimension 0 for all elements of `args` and `s` is the shape of the array
    returned by `rollout_fn`.
  """
  parameters = jax.tree_leaves(args)[0]
  nlocations = jax.tree_leaves(parameters)[0].shape[0]
  nsamples = jax.tree_leaves(parameters)[0].shape[1]

  location_rngs = jax.random.split(rng, nlocations)

  def rollout_helper(location_rng, location_args):
    sample_rngs = jax.random.split(location_rng, nsamples)
    location_params = location_args[0]

    # Map over parameters and not the rest of args
    return jax.vmap(rollout_fn,
                    [0, 0, None])(sample_rngs, location_params,
                                  location_args[1:])

  return jax.vmap(rollout_helper, [0, 0])(location_rngs, args)


@functools.partial(jax.jit, static_argnums=(0, 4, 5))
def trajectories_from_model(mechanistic_model, parameters, rng,
                            observed_epidemics, length,
                            include_observed):
  """Computes batches of `nsamples` for `parameters` and `observed_epidemics`.

  Args:
    mechanistic_model: a `MechanisticModel`.
    parameters: a batch of parameters accepted by `mechanistic_model`.
    rng: a `jax.random.PRNGKey`.
    observed_epidemics: a batched pytree of observed epidemics that can be
      passed to `mechanistic_model`.
    length: the number of time steps to roll out.
    include_observed: Whether to include observed data into the returned
      trajectories.

  Returns:
    If include_observed is False, an array of shape `[batch, nsamples, length]`
    where `batch` is the size of dimension 0 for all elements of `parameters`.
    Otherwise the length dimension will include the observed data.
  """

  def rollout_fn(rng_, params, args):
    obs = args[0]
    # args is now a list to make compatible with the dynamic model
    return mechanistic_model.predict(
        params, rng_, obs, length, include_observed=include_observed)

  return trajectories(rollout_fn, rng, (parameters, observed_epidemics))


def trajectories_from_dynamic_model(mechanistic_model, parameters, rng,
                                    observed_epidemics, dynamic_covariates):
  """Computes batches of `nsamples` for `parameters` and `observed_epidemics`.

  Args:
    mechanistic_model: a `MechanisticModel`.
    parameters: a batch of parameters accepted by `mechanistic_model`.
    rng: a `jax.random.PRNGKey`.
    observed_epidemics: a batched pytree of observed epidemics that can be
      passed to `mechanistic_model`.
    dynamic_covariates: The dynamic_covariates behind the full time course --
      i.e. starting with the observed data's dynamic_covariates and ending
      however far into the "future" we wish to predict.

  Returns:
    An array of shape`[batch, nsamples, dynamic_covariates.sizes['time']]`.
  """
  jnp_dynamic_covariates = jnp.asarray(
      dynamic_covariates.transpose('location', 'time',
                                   'dynamic_covariate').data)
  return _trajectories_from_dynamic_model_helper(mechanistic_model, parameters,
                                                 rng, observed_epidemics,
                                                 jnp_dynamic_covariates)


@functools.partial(jax.jit, static_argnums=(0,))
def _trajectories_from_dynamic_model_helper(mechanistic_model, parameters, rng,
                                            observed_epidemics,
                                            jnp_dynamic_covariates):
  """Computes batches of `nsamples` for `parameters` and `observed_epidemics`.

  Args:
    mechanistic_model: a `MechanisticModel`.
    parameters: a batch of parameters accepted by `mechanistic_model`.
    rng: a `jax.random.PRNGKey`.
    observed_epidemics: a batched pytree of observed epidemics that can be
      passed to `mechanistic_model`.
    jnp_dynamic_covariates: The dynamic_covariates behind the full time
      course -- i.e. starting with the observed data's dynamic_covariates and
      ending however far into the "future" we wish to predict.

  Returns:
    An array of shape`[batch, nsamples, dynamic_covariates.sizes['time']]`.
  """
  def rollout_fn(rng_, params, args):
    obs, dynamic_covariates_slice = args
    return mechanistic_model.predict(
        params, rng_, obs, dynamic_covariates_slice, include_observed=True)

  return trajectories(rollout_fn, rng,
                      (parameters, observed_epidemics, jnp_dynamic_covariates))
