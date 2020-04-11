"""### flatten_util for tf"""

# Implement a tf-compatible ravel_pytree, similar to jax.flatten_util.ravel_pytree.
# Built off of jax's tree_util.
import functools
import tensorflow as tf
from jax import tree_util


def _slice_and_reshape(offset, next_offset, new_shape, flat):
  return tf.reshape(flat[offset:next_offset], new_shape)


def _make_flat_and_unravel_list(leaves):
  flat_accum = []
  unravel_accum = []
  offset = 0
  for leaf in leaves:
    flat_leaf = tf.reshape(leaf, (-1,))
    flat_accum.append(flat_leaf)
    leaf_shape = tuple(leaf.shape.as_list())
    next_offset = offset + flat_leaf.shape.as_list()[0]
    unravel_accum.append(
        functools.partial(_slice_and_reshape, offset, next_offset, leaf_shape))
    offset = next_offset

  def unravel_list(flat):
    accum = []
    for unravel in unravel_accum:
      accum.append(unravel(flat))
    return accum

  return tf.concat(flat_accum, axis=0), unravel_list


def ravel_pytree(pytree):
  leaves, treedef = tree_util.tree_flatten(pytree)
  flat, unravel_list = _make_flat_and_unravel_list(leaves)
  unravel_pytree = lambda flat: treedef.unflatten(unravel_list(flat))
  return flat, unravel_pytree
