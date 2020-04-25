# Lint as: python3
"""A model that uses Bayesian Information Criterion to select parameters."""

import collections

from typing import Optional, Union

import dataclasses

from epi_forecast_stat_mech.statistical_models import probability
from epi_forecast_stat_mech.statistical_models import tree_util

import jax
import jax.numpy as jnp
import numpy as np

import tensorflow_probability
tfp = tensorflow_probability.experimental.substrates.jax
tfd = tfp.distributions

Array = Union[jnp.DeviceArray, np.ndarray, float]


class LinearParameters(collections.namedtuple(
    'LinearModelParameters',
    ('alpha',  # An array of linear weights.
     'intercept',  # The intercept of a linear model.
     ))):
  """Parameters for `LinearModel`."""

  @classmethod
  def init(cls, covariate_dim, prediction_dim, intercept=None):
    # TODO(jamieas): initialize more intelligently.
    if intercept is None:
      intercept = jnp.zeros([1, prediction_dim])
    return cls(alpha=jnp.zeros([covariate_dim, prediction_dim]),
               intercept=intercept)


@dataclasses.dataclass
class LinearModel:
  """A linear model for predicting mechanistic parameters."""

  error_scale: Array = 1.
  # The scale of the Laplace prior placed on the parameters. A larger value
  # corresponds to a less informative prior. If `None`, then we use an improper
  # flat prior (effectively scale=infinity).
  # TODO(jamieas): make it easier to pass per-parameter scales so that we can
  # construct a data-dependent prior.
  laplace_prior_scale: Optional[Array] = None

  def log_prior(self, parameters):
    """Returns the log prior probability of `parameters`."""
    if self.laplace_prior_scale is None:
      return 0.
    return jax.tree_map(probability.soft_laplace_log_prob, parameters)

  def log_likelihood(self, parameters, covariates, mechanistic_parameters):
    """Returns the log likelihood of `mechanistic_parameters`."""
    predicted_mech_params = self.predict(parameters, covariates)
    gaussian = tfd.Normal(predicted_mech_params, self.error_scale)
    mech_params_array, _ = tree_util.pack(mechanistic_parameters)
    return gaussian.log_prob(mech_params_array)

  def predict(self, parameters, covariates):
    """Predicts `mechanistic_params` given `covariates` and `parameters`."""
    covariates_array, _ = tree_util.pack(covariates)
    return covariates_array.dot(parameters.alpha) + parameters.intercept
