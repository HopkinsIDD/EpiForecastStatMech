# Lint as: python3
"""Functions for manipulating nested structures of arrays."""

import functools
from typing import Any, Union
import jax
import jax.numpy as jnp


def slice_along_axis(
    inputs: Any,
    axis: int,
    idx: Union[slice, int]
):
  """Returns slice of `inputs` defined by `idx` along axis `axis`.

  Args:
    inputs: pytree of arrays to slice.
    axis: axis along which to slice the `inputs`.
    idx: index or slice along axis `axis` that is returned.

  Returns:
    Slice of `inputs` defined by `idx` along axis `axis`.
  """
  leaves, tree_def = jax.tree_flatten(inputs)
  sliced = []
  for array in leaves:
    ndim = array.ndim
    slc = tuple(idx if j == axis else slice(None, None) for j in range(ndim))
    sliced.append(array[slc])
  return jax.tree_unflatten(tree_def, sliced)


def split_along_axis(
    inputs: Any,
    axis: int,
    split_idx: int,
    post_squeeze_first: bool = True,
    post_squeeze_second: bool = False
):
  """Returns a tuple of slices of `inputs` split along `axis` at `split_idx`.

  Args:
    inputs: pytree of arrays to split.
    axis: axis along which to split the `inputs`.
    split_idx: index along axis `axis` representing the first element in the
      second split.
    post_squeeze_first: whether to squeeze first slice along the `axis`.
    post_squeeze_second: whether to squeeze second slice along the `axis`.

  Returns:
    Tuple of slices of `inputs` split along `axis` at `split_idx`.
  """
  first_slice = slice_along_axis(inputs, axis, slice(0, split_idx))
  second_slice = slice_along_axis(inputs, axis, slice(split_idx, None))
  squeeze_fn = functools.partial(jnp.squeeze, axis=axis)
  if post_squeeze_first:
    first_slice = jax.tree_map(squeeze_fn, first_slice)
  if post_squeeze_second:
    second_slice = jax.tree_map(squeeze_fn, second_slice)
  return first_slice, second_slice
