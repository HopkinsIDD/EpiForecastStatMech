# Lint as: python3
"""Pull the various high_level models into a single module."""

from epi_forecast_stat_mech import iterative_estimator
from epi_forecast_stat_mech import rtlive
from epi_forecast_stat_mech import sparse_estimator
from epi_forecast_stat_mech import stat_mech_estimator

from epi_forecast_stat_mech.iterative_estimator import IterativeEstimator
from epi_forecast_stat_mech.rtlive import RtLiveEstimator
from epi_forecast_stat_mech.sparse_estimator import SparseEstimator
from epi_forecast_stat_mech.stat_mech_estimator import StatMechEstimator


def get_estimator_dict():
  estimator_dict = {}
  modules = [rtlive, iterative_estimator, sparse_estimator, stat_mech_estimator]
  for module in modules:
    estimator_dict.update(module.get_estimator_dict())
  return estimator_dict
