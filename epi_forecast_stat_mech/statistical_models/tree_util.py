# Lint as: python3
"""Functions for maniputaling nested structures of arrays."""

import jax.numpy as jnp
import numpy as np
import tree


def tree_flatten(pytree):
  """Flattens a pytree.

  Args:
    pytree: a pytree to flatten.
  Returns:
    a pair where the first element is a list of leaf-values and the second is
    the original pytree to use when unflattening.
  """
  return tree.flatten(pytree), pytree


def tree_unflatten(treedef, leaves):
  """Reconstructs a pytree from the treedef and the leaves.

  Args:
    treedef: a treedef to reconstruct
    leaves: a list of leaves used to reconstruct the pytree.
  Returns:
    a reconstructed pytree with leaves `leaves` and tree structure `treedef`.
  """
  return tree.unflatten_as(treedef, leaves)


def tree_map(f, structure):
  """Maps a function across a tree structure.

  Args:
    f: function to be mapped.
    structure: a tree structure to be mapped over.
  Returns:
    a tree structure containing the result of the mapped function.
  """
  return tree.map_structure(f, structure)


def tree_multimap(f, structure, *rest):
  """Maps a function across multiple tree structures.

  Args:
    f: function to be mapped.
    structure: a tree structure to be mapped over.
    *rest: a tuple of tree structures to be mapped over.
  Returns:
    a tree structure containing the result of the mapped function.
  """
  return tree.map_structure(f, structure, *rest)


def pack(pytree, axis=-1):
  """Packs `tree` by concatenating leaves along `axis`.

  Args:
    pytree: pytree to be packed.
    axis: axis along which to leaves are concatenated.

  Returns:
    array representing a packed `tree` and `unpack` function.
  """
  flat, treedef = tree_flatten(pytree)
  splits = np.cumsum(np.array([f.shape[axis] for f in flat]))[:-1]
  packed = jnp.concatenate(flat, axis)
  def unpack(array):
    split = jnp.split(array, splits, axis)
    return tree_unflatten(treedef, split)
  return packed, unpack
