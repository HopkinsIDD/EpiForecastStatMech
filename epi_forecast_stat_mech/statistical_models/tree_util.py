# Lint as: python3
"""Functions for maniputaling nested structures of arrays."""

import jax
import jax.numpy as jnp
import numpy as np


def pack(tree, axis=-1):
  """Packs `tree` by concatenating leaves along `axis`.

  Args:
    tree: pytree to be packed.
    axis: axis along which to leaves are concatenated.

  Returns:
    array representing a packed `tree` and `unpack` function.
  """
  flat, treedef = jax.tree_flatten(tree)
  splits = np.cumsum(np.array([f.shape[axis] for f in flat]))[:-1]
  packed = jnp.concatenate(flat, axis)
  def unpack(array):
    split = jnp.split(array, splits, axis)
    return jax.tree_unflatten(treedef, split)
  return packed, unpack
