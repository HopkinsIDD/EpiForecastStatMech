# Lint as: python3
"""A high level modeling interface for wrapping models & fitting procedures."""
from typing import TypeVar
import xarray

T = TypeVar("T")


class Estimator:
  """Abstract base class for the high level modeling interface.

  An implementation of Estimator wraps statistical and/or mechanistic
  epidemic models along with a fitting procedure to real-world or simulated
  data.

  The design is loosely based on the Scikit-Learn Esimator API, except argument
  and returns values are in the form of labeled ``xarray.Dataset`` objects. The
  idea is that a uniform interface should make it easier to run high level
  evaluations.

  Key methods for `Estimator` subclasses:
    __init__: the signature of ``__init__`` is allowed to vary between models.
      This is where any model specific hyperparameters should be set.
    fit: fit this estimator to data.
    predict: make predictions based on the fit estimator.

  TODO(shoyer): write down lists of canonical variable names that should/could
  appear in ``observations``, ``covariates`` and ``predictions``.
  """

  def fit(self: T, observations: xarray.Dataset) -> T:  # pylint: disable=unused-argument
    """Fit this Estimator based on observations.

    A typical ``fit`` method might do something like the following:
    1. Convert ``observations`` from xarray.Dataset objects into "design
       matrices" of NumPy array(s).
    2. Run some sort of optimization/fitting procedure to minimize the loss of
       statistical and/or mechanicistic models.
    3. Save optimal parameters on the estimator object.

    Args:
      observations: a dataset of observed variables, including (at least?)
        dimensions "location" and "time". "time" should be real-world time
        (i.e., datetime64 dtype), not time since the start of the epidemic.

    Returns:
      This estimator object.
    """
    return self

  def predict(self, test_data, num_samples, seed):
    """Make predictions from this estimator.

    The implementation of typical ``predict`` method would sample from the
    same underlying probabilistic models used in the ``fit`` step.

    The behavior of `predict` is not well defined if `fit` has not been called
    first.

    Args:
      test_data: an xarray dataset with dynamic covariates. The time attribute
        of test_data specifies how long to roll out trajectories.
      num_samples: the number of trajectories to generate for each location.
      seed: an integer seed for generating rollouts.

    Returns:
      An xarray of predictions. The axes are `location`, `sample` and `time`.
    """
    raise NotImplementedError
