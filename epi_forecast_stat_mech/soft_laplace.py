"""### SoftLaplace Model

$$\varphi(x) = \exp\left(-\left(\sqrt{x^2 + 1} - 1\right) - d\right)$$
$$d \approx 1.185495232349193$$

$$\varphi_{m,s}(x) = \varphi\left(\frac{x - m}{s}\right) / s$$
$$\frac{d}{dx} \log \varphi_{m,s}(x) = \left[\frac{d}{du} -\sqrt{u^2 + 1}\right] \left[\frac{d}{dx} \frac{x - m}{s}\right] = -s^{-1} \frac{u}{\sqrt{u^2 + 1}} \approx -s^{-1} \mathop{sign}(u)$$, for $u = \frac{x - m}{s}$, as $|u| >> 1$.


Impose $K\varphi_{m,s}(0)=c_0 \approx 1$.
* $\varphi(-m/s) = c_0 s / K$
* $\exp\left(-\left(\sqrt{m^2/s^2 + 1} - 1\right) - d\right) = c_0 s / K$
* $-\left(\sqrt{m^2/s^2 + 1} - 1\right) - d = \log(c_0 s / K)$
* $\sqrt{m^2/s^2 + 1} - 1  = -(\log(c_0 s / K) + d)$
* $\sqrt{m^2/s^2 + 1}  = -(\log(c_0 s / K) + d - 1)$
* $m^2/s^2 + 1  = (\log(c_0 s / K) + d - 1)^2$
* $m^2  = s^2[(\log(c_0 s / K) + d - 1)^2 - 1]$
* $m  = \pm s\sqrt{(\log(c_0 s / K) + d - 1)^2 - 1}$
* $m  = \pm s\sqrt{(\log(K / (c_0 s)) + 1 - d)^2 - 1}$
* $m \approx s |\log(K / (c_0 s))|$ (drop the abs when $K> c_0 s$ is safe).

Conclusion: 
* $r = 1/s$
* One can eliminate $m$ by choosing an initial intensity $c_0$ and enforcing $K\varphi_{m,s}(0)=c_0$, which leads to $m = s\sqrt{(\log(K / (c_0 s)) + 1 - d)^2 - 1}$.
"""

from .intensity_family import IntensityFamily
from .tf_common import *


class SoftLaplaceParams(object):

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
    return 'SoftLaplaceParams(m={m}, s={s}, K={K})'.format(
        m=self.m, s=self.s, K=self.K)

  def __repr__(self):
    return 'SoftLaplaceParams({x})'.format(x=list(self._x))


def e_half_loss(x):
  return tf.sqrt(x**2 + 1.) - 1.


def soft_laplace_logprob_base(x):
  return -e_half_loss(x) - 1.185495232349193


def shift_scaleify_logprob(base_logprob):
  return lambda m, s: lambda x: base_logprob((x - m) / s) - tf.math.log(s)


soft_laplace_logprob = shift_scaleify_logprob(soft_laplace_logprob_base)


def basic_simpson(f, t0, t1):
  t_mid = (t1 + t0) / 2.
  h = (t1 - t0) / 2.
  return h / 3. * (f(t0) + 4. * f(t_mid) + f(t1))


def soft_laplace_parametric_curve_base(t, m, s, K):
  f = lambda t: K * tf.exp(soft_laplace_logprob(m, s)(t))
  return f(t - .5) * 1.


def soft_laplace_parametric_curve_simpson(t, m, s, K):
  f = lambda t: K * tf.exp(soft_laplace_logprob(m, s)(t))
  return basic_simpson(f, t - 1., t)


def soft_laplace_intensity_core(t, m, s, K):
  preds = soft_laplace_parametric_curve_base(t, m, s, K)
  preds = tf.math.maximum(preds, 0.1)
  return preds


def soft_laplace_intensity(trajectory, params):
  preds = soft_laplace_intensity_core(trajectory.t, params.m, params.s,
                                      params.K)
  return preds


# params0 = SoftLaplaceParams().init(m=100., s=300., K=1000.)
params0 = SoftLaplaceParams().init(m=12.0, s=1.4, K=5000.)

SoftLaplaceFamily = IntensityFamily(
    name='SoftLaplace',
    intensity=soft_laplace_intensity,
    params_wrapper=SoftLaplaceParams,
    params0=params0,
    param_names=['m', 's', 'K'],
    encoded_param_names=['m', 'log_s', 'log_K'])
