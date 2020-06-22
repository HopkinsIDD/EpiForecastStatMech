# Lint as: python3
"""A few methods to handle NaNs.

Library functions that take a numpy array.
Assumes dimensions are (location, time).
"""
import itertools
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.ndimage


def _assume_2d(array):
  if len(array.shape) != 2:
    raise ValueError(
        'Functions in this library assume a 2d array with dimensions (location, time).'
    )


def fillna_zero(array):
  """Fills all nans with zero."""
  return np.nan_to_num(array)


def median_filter(array, filter_days=5):
  """Median filter.

  Doesn't guarantee all nans will be filled.

  Args:
    array: A 2d numpy array with dimensions (location, time)
    filter_days: An odd integer >= 1, size of the filter.
  Returns:
    A numpy array.
  """
  _assume_2d(array)
  return scipy.ndimage.generic_filter(
      array, np.nanmedian, size=(1, filter_days), mode='nearest')


def mean_filter(array, filter_days=5):
  """Mean filter.

  Doesn't guarantee all nans will be filled.

  Args:
    array: A 2d numpy array with dimensions (location, time)
    filter_days: An odd integer >= 1, size of the filter.
  Returns:
    A numpy array.
  """
  _assume_2d(array)
  return scipy.ndimage.generic_filter(
      array, np.nanmean, size=(1, filter_days), mode='nearest')


def fillna_beginning(array, value=0):
  """Fills consecutive NaNs at the beginning of an array.

  Args:
    array: A 2d numpy array with dimensions (location, time)
    value: Float, the value to fill.
  Returns:
    A numpy array.
  """

  def fillna_beginning_1d(array):
    array = array.copy()
    num_initial_nans = 0
    while np.isnan(array[num_initial_nans]):
      num_initial_nans += 1
      if num_initial_nans == len(array):
        break
    array[:num_initial_nans] = value
    return array

  _assume_2d(array)
  return np.apply_along_axis(fillna_beginning_1d, 1, array)


def fillna_ffill(array):
  """Forward fills an array.

  Args:
    array: A 2d numpy array with dimensions (location, time)
  Returns:
    A numpy array.
  """

  _assume_2d(array)
  mask = np.isnan(array)
  idx = np.where(~mask, np.arange(mask.shape[1]), 0)
  max_idx = np.maximum.accumulate(idx, axis=1)
  result = array[np.arange(max_idx.shape[0])[:, None], max_idx]
  return result


def fillna_bfill(array):
  """Backward fills an array.

  Args:
    array: A 2d numpy array with dimensions (location, time)
  Returns:
    A numpy array.
  """

  _assume_2d(array)
  flipped = np.flip(array)
  filled = fillna_ffill(flipped)
  return np.flip(filled)


def fillna_interp(array):
  """Applies 1-d linear interpolation.

  Args:
    array: A 2d numpy array with dimensions (location, time)
  Returns:
    A numpy array.
  """

  def interp_1d(array):
    indices = np.arange(array.shape[0])
    values = np.where(np.isfinite(array))
    # Interp requires at least 2 non-nan entries
    if len(values[0]) < 2:
      return array
    f = scipy.interpolate.interp1d(
        indices[values], array[values], bounds_error=False)
    return np.where(np.isfinite(array), array, f(indices))

  _assume_2d(array)
  return np.apply_along_axis(interp_1d, 1, array)


def plot_nans(array, title, ax=None):
  """Plots nans."""
  if ax is None:
    plt.figure()
    ax = plt.gca()

  ax.imshow(np.isnan(array), interpolation='nearest')
  ax.set_xlabel('time')
  ax.set_ylabel('location')
  ax.set_title(title)


def longest_nans(array):
  """Returns the longest consecutive nans in an array.

  Args:
    array: A 2d numpy array with dimensions (location, time)
  Returns:
    A numpy array.
  """

  def longest_nans_1d(array):
    mask = np.isnan(array)
    grouped = [(el, sum(1
                        for element in group))
               for el, group in itertools.groupby(mask)]
    nan_groups = [g[1] for g in grouped if g[0] == 1]
    if not nan_groups:
      return 0
    return max(nan_groups)

  _assume_2d(array)
  return np.apply_along_axis(longest_nans_1d, 1, array)
