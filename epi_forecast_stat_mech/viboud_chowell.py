r"""### Viboud Chowell Model


This model states that if the infected number of infections at time $t$ is given by $$E(I_t) = r Y_t^p (1-Y_t/Y_\infty)^a, $$

maybe revise to this:
$$E(I_t) = r Y_t^p (1-(Y_t/Y_\infty)^a), $$

where $r,p,a$ are prameters of the model and $Y_\infty$ (aka $K$) is the total number of infections in the epidemic.

For implementation, use:
$$E(I_t) = \max(r Y_t^p (\max(1-Y_t/Y_\infty, 1E-6))^a, 0.1). $$
(The inner max avoids nan-issues; the outer keeps the prediction positive).
"""

from .intensity_family import IntensityFamily
from .tf_common import *


epsilon = np.finfo('float32').tiny


expit = tf.math.sigmoid


def logit(p):
  return tf.math.log(p / (1. - p))

def decode_for_a(x):
  return expit(x) * .5 + .5

def encode_a(a):
  return logit(2. * (a - .5))


class ViboudChowellParams(object):

  def __init__(self):
    self._x = None

  def reset(self, x):
    self._x = tf_float(x)
    return self

  def init(self, r, a, p, K):
    r = tf_float(r)
    a = tf_float(a)
    p = tf_float(p)
    K = tf_float(K)
    self.reset(
        tf.stack(
            [tf.math.log(r),
             encode_a(a),
             tf.math.log(p),
             tf.math.log(K)]))
    return self

  @property
  def r(self):
    return tf.exp(self._x[0])

  @property
  def a(self):
    return decode_for_a(self._x[1])

  @property
  def p(self):
    return tf.exp(self._x[2])

  @property
  def K(self):
    return tf.exp(self._x[3])

  def __str__(self):
    return 'ViboudChowellParams(r={r}, a={a}, p={p}, K={K})'.format(
        r=self.r, a=self.a, p=self.p, K=self.K)


def viboud_chowell_intensity_core(y, r, a, p, K):
  # y is the total prior cases
  preds = tf.math.maximum(r * y**p * (tf.math.maximum(1 - y / K, epsilon))**a, 0.1)
  return preds


def viboud_chowell_intensity(trajectory, vc_params):
  previous_day_cumsum = np.concatenate([
      np_float([0.]),
      np.cumsum(np_float(trajectory.new_infections))[:-1]])
  preds = viboud_chowell_intensity_core(
      previous_day_cumsum, vc_params.r, vc_params.a,
      vc_params.p, vc_params.K)
  return preds


params0 = ViboudChowellParams().init(r=2., a=.9, p=.9, K=2000.)

ViboudChowellFamily = IntensityFamily(
    name='Viboud-Chowell',
    intensity=viboud_chowell_intensity,
    params_wrapper=ViboudChowellParams,
    params0=params0,
    param_names=['r', 'a', 'p', 'K'],
    encoded_param_names=['log_r', 'logit_scaled_a', 'log_p', 'log_K'])
