# Lint as: python3
"""Functions to return constant values for epi_forecast_stat_mech.
"""
import numpy as np


def coord_units(key):
  return {
      'sample': 'arbitrary int',
      'location': 'arbitrary int',
      'time': 'simulation step',
      'static_covariate': 'arbitrary int',
  }.get(key)


def coordinates(key, number):
  return {
      'sample': np.arange(number),
      'location': np.arange(number),
      'time': np.arange(number),
      'static_covariate': np.arange(number),
  }.get(key)
