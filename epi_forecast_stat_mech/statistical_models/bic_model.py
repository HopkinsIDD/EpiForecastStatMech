# Lint as: python3
"""A model that uses Bayesian Information Criterion to select parameters."""

from typing import Any

import dataclasses

from epi_forecast_stat_mech.statistical_models import tree_util

import jax
import jax.numpy as jnp


def logistic(x):
  return 1 / (1 + jnp.exp(-x))


def soft_nonzero(x, sharpness, threshold):
  """Returns an approximate equivalent to `(jnp.abs(x) > 0) * 1`."""
  return logistic(sharpness * (jnp.abs(x) - threshold))


def bic_adjustment(num_data_points, num_parameters):
  """Returns the adjustment due to Bayesian information criteria.

  We take some liberties with the definition of BIC. See
  https://en.wikipedia.org/wiki/Bayesian_information_criterion. The adjustment
  should be performed as follows:

  ```
    bic_log_likelihood = model_log_likelihood + bic_adjustment(...)
  ```

  Args:
    num_data_points: the number of observations.
    num_parameters: the number of parameters.

  Returns:
    The adjustment to the log likelihood due to BIC.
  """
  return - jnp.log(num_data_points) * num_parameters / 2


@dataclasses.dataclass
class BICModel:
  """A wrapper that applies Bayesian Information Criterion to a model."""

  base_model: Any
  bic_multiplier: float = 1.
  nonzero_threshold: float = .1
  nonzero_sharpness: float = 20

  def log_prior(self, parameters):
    return self.base_model.log_prior(parameters)

  def _count_nonzero(self, params):
    param_list, _ = tree_util.tree_flatten(params)
    return sum([
        soft_nonzero(p, self.nonzero_sharpness, self.nonzero_threshold).sum()
        for p in param_list])

  def log_likelihood(self, parameters, covariates, mechanistic_parameters):
    """Returns the log likelihood of `mechanistic_parameters`."""
    base_log_likelihood = self.base_model.log_likelihood(parameters,
                                                         covariates,
                                                         mechanistic_parameters)
    # `covariate_matrix` has shape `[num_epidemics, covariate_dim]`.
    covariate_matrix, _ = tree_util.pack(covariates)
    num_data_points = covariate_matrix.shape[0]
    num_parameters = self._count_nonzero(parameters)
    return base_log_likelihood + bic_adjustment(num_data_points, num_parameters)

  def predict(self, parameters, covariates):
    return self.base_model.predict(parameters, covariates)
