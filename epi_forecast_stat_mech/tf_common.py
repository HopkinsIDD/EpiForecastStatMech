import numpy as np
import tensorflow as tf
import scipy.optimize


def tf_float(x):
  return tf.convert_to_tensor(x, dtype=tf.float32)


def np_float(x):
  return np.asarray(x, dtype=np.float64)


def tf_float_star(val):
  if isinstance(val, tuple):
    return tuple(tf_float_star(u) for u in val)
  if isinstance(val, list):
    return [tf_float_star(u) for u in val]
  return tf_float(val)


def np_float_star(val):
  if isinstance(val, tuple):
    return tuple(np_float_star(u) for u in val)
  if isinstance(val, list):
    return [np_float_star(u) for u in val]
  return np_float(val)


def tf_to_np_wrap(tf_fun, tf_float, np_float):

  def wrapped(*pargs):
    return np_float_star(tf_fun(*tf_float_star(pargs)))

  return wrapped


def wrap_minimize(tf_fun, x0, **kwargs):
  fun = tf_to_np_wrap(tf_fun, tf_float, np_float)
  return scipy.optimize.minimize(fun=fun, x0=np_float(x0), **kwargs)
