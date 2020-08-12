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

  def mask_to_recent(self, recent_day_limit):
    """Further restrict the current mask to the most recent passing values."""
    days_left = self.mask[:, ::-1].cumsum(axis=-1)[:, ::-1]
    mask = self.mask & (days_left <= recent_day_limit)
    self.mask = mask

  def mask_dynamic_cov(self, data, cov_name, mask_value):
    """Masks times when cov_data does not have mask_value."""
    cov_data = data.dynamic_covariates.sel(
        dynamic_covariate=cov_name, drop=True)
    dynamic_cov = cov_data.transpose("location", "time")
    mask = np.asarray(self.mask & (dynamic_cov == mask_value))
    self.mask = mask


def make_mask(data,
              min_value=None,
              recent_day_limit=None,
              dynamic_cov_name=None,
              dynamic_cov_value=0):
  """Function to make masks. Defaults to just non-null values.

  The masking operations, if not null are applied in the order of
  non-nan (at construction), min_infections, recent_day_limit, dynamic_cov.
  Not all combinations are necessarily sensible.

  Arguments:
    data: An xarray passing validate_data.
    min_value: Mask away location, time for which the cumulative infections
      are strictly below min_value.
    recent_day_limit: Mask away location, time that are not among the most
      recent (non-masked) events for that location.
    dynamic_cov_name: Named dynamic_covariate to use in a dynamic_cov-based
      filter.
    dynamic_cov_value: mask to location, time where the named dynamic_cov
      takes this value.

  Returns:
    An np.float32 array of location, time with 1. or 0. values.
  """
  timemask = MaskTime(data)
  if min_value is not None:
    timemask.mask_min_infections(min_value)
  if recent_day_limit is not None:
    timemask.mask_to_recent(recent_day_limit)
  if dynamic_cov_name is not None:
    timemask.mask_dynamic_cov(data, dynamic_cov_name, dynamic_cov_value)
  return timemask.mask.astype(np.float32)
