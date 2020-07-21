"""Class to mask parts of infection curves in time."""

import numpy as np


class MaskTime():
  """A class to mask/keep parts of infection curves in time."""

  def __init__(self, data):
    self.new_infections = data.new_infections.transpose("location", "time")
    self.mask = np.asarray(~self.new_infections.isnull())

  def mask_min_infections(self, min_value):
    """Masks times while total number of infections is less than `min_value`."""
    total = self.new_infections.cumsum("time", skipna=True)
    mask = np.asarray((total >= min_value) & self.mask)
    self.mask = mask

  def mask_dynamic_cov(self, data, cov_name, mask_value):
    """Masks times when cov_data does not have mask_value."""
    cov_data = data.dynamic_covariates.sel(dynamic_covariate=cov_name, drop=True)
    dynamic_cov = cov_data.transpose("location", "time")
    mask = np.asarray(self.mask & (dynamic_cov == mask_value))
    self.mask = mask


def make_mask(data, min_value=None, dynamic_cov_name=None,
              dynamic_cov_value=0):
  """Function to make masks. Defaults to just non-null values."""
  timemask = MaskTime(data)
  if min_value is not None:
    timemask.mask_min_infections(min_value)
  if dynamic_cov_name is not None:
    timemask.mask_dynamic_cov(data, dynamic_cov_name, dynamic_cov_value)
  return timemask.mask.astype(np.float32)
