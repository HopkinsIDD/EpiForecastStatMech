"""Abstract class to represent predictable properties of a mech_model."""
import abc
from jax import numpy as jnp


class Observables(object):
  """Computes statistical observables of the epidemics."""

  @abc.abstractmethod
  def observables(self, mech_model, mech_params, epidemic):
    """Computes statistical observables of the epidemic.

    Could be any set of values that represent global properties of the epidemics
    e.g. total size of the epidemics or peak location.

    Args:
      mech_model: A MechanisticModel instance.
      mech_params: Parameters of the mechanistic model (one location's worth).
      epidemic: Observed epidemic trajectory (one location's worth).

    Returns:
      Pytree holding the estimates of statistical properties of
      the epidemic based on the mechanistic model. This can vary between all
      model parameters to a subset or other statistical predictions by the
      model.
    """
    ...


class InternalParams(Observables):

  def observables(self, mech_model, mech_params, epidemic):
    return dict(zip(mech_model.encoded_param_names,
                    jnp.split(mech_params, len(mech_params))))


class ObserveSpecified(Observables):

  def __init__(self, specified_internal_params):
    self._specified_internal_params = specified_internal_params

  def observables(self, mech_model, mech_params, epidemic):
    encoded_dict = InternalParams().observables(mech_model, mech_params,
                                                epidemic)
    return {key: encoded_dict[key] for key in self._specified_internal_params}
