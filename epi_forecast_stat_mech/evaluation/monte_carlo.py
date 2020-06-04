# Lint as: python3
"""Code for drawing Monte Carlo rollouts from models."""

import functools

import jax


def trajectories(rollout_fn, rng, args, nlocations, nsamples):
  """Computes batches of `nsamples` trajectories for all `args`.

  Args:
    rollout_fn: a callable that takes `(rng, args)` and returns an array.
    rng: a `jax.random.PRNGKey`.
    args: a pytree of `args` that can be passed to `rollout_fn`. All args will
      be vmapped along axis 0.
    nsamples: the number of samples for each 'row' of `args`.

  Returns:
    An array of shape `[batch, nsamples] + s` where `batch` is the size of
    dimension 0 for all elements of `args` and `s` is the shape of the array
    returned by `rollout_fn`.
  """
  rngs = jax.random.split(rng, nlocations)

  def rollout_helper(rng, args):
    rngs = jax.random.split(rng, nsamples)
    return jax.vmap(rollout_fn, [0, None])(rngs, args)

  return jax.vmap(rollout_helper, [0, 0])(rngs, args)


@functools.partial(jax.jit, static_argnums=(0, 4, 5, 6))
def trajectories_from_model(mechanistic_model, parameters, rng,
                            observed_epidemics, length, nsamples,
                            include_observed):
  """Computes batches of `nsamples` for `parameters` and `observed_epidemics`.

  Args:
    mechanistic_model: a `MechanisticModel`.
    parameters: a batch of parameters accepted by `mechanistic_model`.
    rng: a `jax.random.PRNGKey`.
    observed_epidemics: a batched pytree of observed epidemics that can be
      passed to `mechanistic_model`.
    length: the number of time steps to roll out.
    nsamples: the number of samples for each 'row' of `parameters` and
      `observations`.
    include_observed: Whether to include observed data into the returned
      trajectories.

  Returns:
    If include_observed is False, an array of shape `[batch, nsamples, length]`
    where `batch` is the size of dimension 0 for all elements of `parameters`.
    Otherwise the length dimension will include the observed data.
  """

  nlocations = jax.tree_leaves(parameters)[0].shape[0]

  def rollout_fn(rng_, args):
    params, obs = args
    return mechanistic_model.predict(
        params, rng_, obs, length, include_observed=include_observed)

  return trajectories(rollout_fn, rng, (parameters, observed_epidemics),
                      nlocations, nsamples)