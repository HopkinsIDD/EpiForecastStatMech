# Lint as: python3
"""Statistical models for forecasting epidemics."""

import abc


class StatisticalModel:
  """Abstract class representing statistical models."""

  @abc.abstractmethod
  def init_parameters(self):
    """Returns initial parameters generated at model construction time."""
    ...

  @abc.abstractmethod
  def log_prior(self, parameters):
    """Returns the log probability of `parameters` based on prior assumptions.

    Args:
      parameters: parameters of the statistical model.

    Returns:
      log-probabilities for each parameter. Must be of the same structure as
      `parameters`.
    """
    ...

  @abc.abstractmethod
  def log_likelihood(self, parameters, covariates, observations):
    """Returns the log likelihood of `observations` given the `covariates`.

    Args:
      parameters: parameters of the statistical model.
      covariates: array representing covariates for each location. Locations
        should be represented by the leading axis.
      observations: observations of epidemics for each location for which log
        likelihood is predicted. Can be mechanistic model parameters, subset
        or other derived quantities that are characteristics of the epidemic.
        Leading dimensions of each pytree-leaf in `observation` must be equal
        to the number of locations.

    Returns:
      pytree of the same. structure as `observations` holding log-likelihoods.
    """
    ...

  @abc.abstractmethod
  def predict(self, parameters, covariates):
    """Predicts a distribution over `observations` based on `covariates`.

    Args:
      parameters: parameters of the statistical model.
      covariates: array representing covariates for each location.

    Returns:
      pytree of probability distributions over `observations` given the
      `parameters` of the statistical model and `covariates`.
    """
    ...
