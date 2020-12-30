"""Construct Adam and BFGS and training loops for a generic objective f.
"""

import functools
import jax
from jax import flatten_util
from jax.experimental import optimizers
import jax.numpy as jnp
import numpy as np
import scipy


def get_adam_optim_loop(f, learning_rate=1E-3):
  """Construct an adam training loop to minimize f."""
  opt_init, opt_update, get_params = optimizers.adam(
      learning_rate, eps=1E-6)

  @jax.jit
  def train_step(step, opt_state):
    params = get_params(opt_state)
    loss_value, grad = f(params)
    opt_state = opt_update(step, grad, opt_state)
    return opt_state, loss_value

  # For some of these models (especially on accelerators), a single training
  # step runs very quickly. Fusing steps together considerably improves
  # performance.
  @functools.partial(jax.jit, static_argnums=(1,))
  def repeated_train_step(step, repeats, opt_state):
    def f(carray, _):
      step, opt_state, _ = carray
      opt_state, loss_value = train_step(step, opt_state)
      return (step + 1, opt_state, loss_value), None
    (_, opt_state, loss_value), _ = jax.lax.scan(
        f, (step, opt_state, 0.0), xs=None, length=repeats)
    return opt_state, loss_value

  def train_loop(x0, train_steps=10000, fused_train_steps=100, verbose=1):
    if verbose >= 2:
      print(f'x0: {x0}')
      print(f'f(x0): {f(x0)}')
    opt_state = opt_init(x0)
    for step in range(0, train_steps, fused_train_steps):
      opt_state, loss_value = repeated_train_step(
          step, fused_train_steps, opt_state)
      if step % 1000 == 0:
        if verbose >= 1:
          print(f'Loss at step {step} is: {loss_value}.')

    x = get_params(opt_state)
    return x
  return train_loop


def np_float(x):
  return np.asarray(x, dtype=np.float64)


def jnp_float(x):
  return jnp.asarray(x, dtype=jnp.float32)


def jnp_float_star(val):
  if isinstance(val, tuple):
    return tuple(jnp_float_star(u) for u in val)
  if isinstance(val, list):
    return [jnp_float_star(u) for u in val]
  return jnp_float(val)


def np_float_star(val):
  if isinstance(val, tuple):
    return tuple(np_float_star(u) for u in val)
  if isinstance(val, list):
    return [np_float_star(u) for u in val]
  return np_float(val)


def jnp_to_np_wrap_val_grad(jnp_val_grad_fun, unravel):

  def wrapped(*pargs):
    pargs2 = jnp_float_star(pargs)
    val, grad = np_float_star(
        jnp_val_grad_fun(*((unravel(pargs2[0]),) + pargs2[1:])))
    flat_grad, _ = flatten_util.ravel_pytree(grad)
    return val, np_float(flat_grad)

  return wrapped


def _wrap_minimize(jnp_fun, x0_in, **kwargs):
  x0, unravel = flatten_util.ravel_pytree(x0_in)
  fun = jnp_to_np_wrap_val_grad(jnp_fun, unravel)
  opt1 = scipy.optimize.minimize(fun=fun, x0=np_float(x0), **kwargs)
  opt_status = (opt1.success, float(opt1.fun), opt1.nfev, opt1.message)
  x = opt1.x
  x_out = unravel(x)
  return x_out, opt_status, opt1


def lbfgs_optim(f, x0, max_iter=10000):
  return _wrap_minimize(
      f,
      x0,
      jac=True,
      method='L-BFGS-B',  # sometimes line-search failure.
      options={'maxiter': max_iter})
