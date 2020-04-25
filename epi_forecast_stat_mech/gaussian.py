"""### Gaussian Model"""
from .intensity_family import IntensityFamily
from .tf_common import *


class GaussianParams(object):

  def __init__(self):
    self._x = None

  def reset(self, x):
    self._x = tf_float(x)
    return self

  def init(self, m, s, K):
    m = tf_float(m)
    s = tf_float(s)
    K = tf_float(K)
    self.reset(tf.stack([m, tf.math.log(s), tf.math.log(K)]))
    return self

  @property
  def m(self):
    return self._x[0]

  @property
  def s(self):
    return tf.exp(self._x[1])

  @property
  def K(self):
    return tf.exp(self._x[2])

  def __str__(self):
    return 'GaussianParams(m={m}, s={s}, K={K})'.format(
        m=self.m, s=self.s, K=self.K)


def gaussian_intensity_core(t, m, s, K):
  dist = tfd.Normal(loc=m, scale=s)
  preds = tf.math.maximum(K * (dist.cdf(t) - dist.cdf(t - 1.)), 0.1)
  return preds


def gaussian_intensity(trajectory, g_params):
  preds = gaussian_intensity_core(
      tf_float(trajectory.time), g_params.m, g_params.s, g_params.K)
  return preds


params0 = GaussianParams().init(m=100., s=100., K=1000.)

GaussianFamily = IntensityFamily(
    name='Gaussian',
    intensity=gaussian_intensity,
    params_wrapper=GaussianParams,
    params0=params0,
    param_names=['m', 's', 'K'],
    encoded_param_names=['m', 'log_s', 'log_K'])
