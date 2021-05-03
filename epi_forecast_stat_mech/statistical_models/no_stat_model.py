"""Null statistical model.

Stat model that always returns 0. This is the null model used to compare
results with and without a statistical model.
"""
from typing import Any, Callable
import dataclasses

from epi_forecast_stat_mech.statistical_models import base


@dataclasses.dataclass
class Null(base.StatisticalModel):
  """Null statistical model.

  Attributes:
    predict_module: flax.nn.Module that takes `inputs` and `output_size`
      arguments and returns array of shape `[batch, output_size]` that will be
      used to predict locations of the gaussian distributed `observations` and
      possibly scales, depending on whether `error_model` is 'full'.
    log_prior_fn: function that computes log_prior on a parameters of the
      `predict_module`.
  """
  log_prior_fn: Callable[..., Any] = None

  def init_parameters(self, rng, covariates, epidemic_observables):
    """Returns initial parameters generated at model construction time."""
    return None

  def log_prior(self, parameters):
    """Returns the log probability of `parameters` based on priors.

    Args:
      parameters: parameters of the statistical model.

    Returns:
      log-probabilities: A scalar
    """
    return 0.

  def log_likelihood(self, parameters, covariates, observations):
    """Returns the log likelihood of `observations`.

    Args:
      parameters: parameters of the statistical model.
      covariates: A numpy array of shape "location" x "static_covariate".
      observations: A tree of mech_params that we want to explain.

    Returns:
      A log-likelihood. The return is a scalar (0).
    """
    return 0.

  def predict(self, parameters, covariates, observations):
    """Predicts a distribution over `observations` based on `covariates`.

    Args:
      parameters: parameters of the statistical model.
      covariates: array representing covariates for each location.
      observations: structure of observations to be predicted.

    Returns:
      None
    """
    return None, None
