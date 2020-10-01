# Lint as: python3
"""Pull the various high_level models into a single module."""

from epi_forecast_stat_mech import ariadne_estimator
from epi_forecast_stat_mech import iterative_estimator
from epi_forecast_stat_mech import iterative_dynamic_estimator
from epi_forecast_stat_mech import rtlive
from epi_forecast_stat_mech import sparse_estimator
from epi_forecast_stat_mech import stat_mech_estimator

from epi_forecast_stat_mech.iterative_estimator import IterativeEstimator
from epi_forecast_stat_mech.iterative_dynamic_estimator import IterativeDynamicEstimator
from epi_forecast_stat_mech.rtlive import RtLiveEstimator
from epi_forecast_stat_mech.sparse_estimator import SparseEstimator
from epi_forecast_stat_mech.stat_mech_estimator import StatMechEstimator


def get_simple_estimator_dict():
  estimator_dict = {}
  modules = [
      rtlive, iterative_estimator, sparse_estimator, stat_mech_estimator,
      iterative_dynamic_estimator
  ]
  for module in modules:
    estimator_dict.update(module.get_estimator_dict())
  return estimator_dict


def get_meta_estimator_dict(validation_times=(14, 28, 42)):
  estimator_dict = {}
  modules = [ariadne_estimator]
  for module in modules:
    estimator_dict.update(module.get_estimator_dict(validation_times))
  return estimator_dict


def get_estimator_dict():
  estimator_dict = get_simple_estimator_dict()
  meta_estimator_dict = get_meta_estimator_dict()
  estimator_dict.update(meta_estimator_dict)
  return estimator_dict
