# Lint as: python3
"""An iterative epi_forecast_stat_mech.estimator_base.Estimator."""
import collections
import functools
import pickle

import jax
from jax import flatten_util
import jax.numpy as jnp
from matplotlib import pyplot as plt
import numpy as np
import scipy
import sklearn
import sklearn.inspection
import xarray

from epi_forecast_stat_mech import data_model
from epi_forecast_stat_mech import estimator_base
from epi_forecast_stat_mech import stat_mech_estimator
from epi_forecast_stat_mech.evaluation import monte_carlo
from epi_forecast_stat_mech.mechanistic_models import mechanistic_models
from epi_forecast_stat_mech.statistical_models import probability as stat_prob


def np_float(x):
  return np.asarray(x, dtype=np.float64)


def jnp_float(x):
  return jnp.asarray(x, dtype=jnp.float32)


def load(file_in):
  return pickle.load(file_in)


def _get_time_mask(data, min_value=1):
  """Masks times while total number of infections is less than `min_value`."""
  new_infections = data.new_infections.transpose('location', 'time')
  total = np.cumsum(new_infections.values, -1)
  mask = np.asarray(total >= min_value & ~new_infections.isnull()).astype(
      np.float32)
  return mask


class IterativeEstimator(estimator_base.Estimator):

  def save(self, file_out):
    pickle.dump(self, file_out)

  def __init__(self, stat_estimators=None):
    if stat_estimators is None:
      stat_estimators = collections.defaultdict(
          lambda: sklearn.ensemble.RandomForestRegressor(n_estimators=50))
    self.stat_estimators = stat_estimators
    self.hat_interpolation_alpha = 0.5
    self.iter_max = 100
    # bake in vc for now
    self.mech_model = mechanistic_models.ViboudChowellModel()
    self.encoded_param_names = ['log_r', 'log_a', 'log_p', 'log_K']
    self.mech_bottom_scale = jnp_float([.1, .1, .1, .1])
    self.out_dim = len(self.encoded_param_names)

  def _unflatten(self, x):
    return jnp.reshape(x, (-1, self.out_dim))

  def fit(self, data):
    data_model.validate_data(data, require_no_samples=True)
    self.data = data
    num_locations = data.sizes['location']
    self.epidemics = epidemics = stat_mech_estimator._pack_epidemics_record_tuple(
        data)
    self.time_mask = time_mask = _get_time_mask(data)
    self.v_df = v_df = _get_static_covariate_df(data)

    def mech_plus_stat_errors(mech_params_stack, mech_params_hat_stack=None):
      mech_log_prior = jnp.sum(
          jax.vmap(self.mech_model.log_prior)(mech_params_stack))
      mech_log_lik_terms_raw = jax.vmap(self.mech_model.log_likelihood)(
          mech_params_stack, epidemics)
      mech_log_lik_terms = jnp.where(time_mask, mech_log_lik_terms_raw, 0.)
      mech_log_lik = jnp.sum(mech_log_lik_terms)
      mech_log_prob = mech_log_prior + mech_log_lik
      mech_log_prob = mech_log_lik
      stat_plugin_error_model = stat_prob.gaussian_with_softplus_scale_estimate(
          mech_params_stack,
          axis=0,
          min_scale=self.mech_bottom_scale,
          mean=mech_params_hat_stack,
          softness=self.mech_bottom_scale)
      # shape: (out_dim,)
      stat_log_prob = stat_plugin_error_model.log_prob(mech_params_stack).sum(
          axis=0)
      mech_plus_current_stat_loss = -(mech_log_prob + jnp.sum(stat_log_prob))
      return mech_plus_current_stat_loss

    mech_plus_stat_loss_val_and_grad = jax.jit(
        jax.value_and_grad(mech_plus_stat_errors))

    mech_params_stack = jnp.stack(
        [self.mech_model.init_parameters() for _ in range(num_locations)])
    assert mech_params_stack.shape[1] == len(self.encoded_param_names)
    mech_params_hat_stack = mech_params_stack

    for _ in range(self.iter_max):
      # Update mech_params_stack "regularized" by current mech_params_hat_stack.
      # N.B. This is not a maximum likelihood update.
      mech_params_stack, opt_status, _ = lbfgs_optim(
          functools.partial(
              mech_plus_stat_loss_val_and_grad,
              mech_params_hat_stack=mech_params_hat_stack), mech_params_stack)
      if not opt_status[0]:
        print('optimizer reports: %s' % (opt_status,))

      # Find an update for mech_params_hat_stack by calling upon the regressors
      # to fit each mech_param. Each of these fits is responsible for it's own
      # internal regularization.
      hat_accum = []
      for j, param_name in enumerate(self.encoded_param_names):
        regressor = self.stat_estimators[param_name]
        regressor.fit(v_df, np_float(mech_params_stack[:, j]))
        hat_accum.append(regressor.predict(v_df))
      proposed_mech_params_hat_stack = jnp.stack(hat_accum, axis=1)

      # To stabilize the iteration, we don't jump all the way to the new fit,
      # but to this convex combination with the old value.
      mech_params_hat_stack = (
          self.hat_interpolation_alpha * proposed_mech_params_hat_stack +
          (1 - self.hat_interpolation_alpha) * mech_params_hat_stack)
    self.mech_params_stack = mech_params_stack
    self.mech_params_hat_stack = mech_params_hat_stack
    return self

  def _check_fitted(self):
    if not hasattr(self, 'mech_params_stack'):
      raise AttributeError('`fit` must be called before `predict`.')

  def plot_partial_dependence(self):
    self._check_fitted()
    v_df = self.v_df
    for _, param_name in enumerate(self.encoded_param_names):
      print('Partial dependence plot for %s' % (param_name,))
      regressor = self.stat_estimators[param_name]
      sklearn.inspection.plot_partial_dependence(
          regressor, v_df, v_df.columns, n_jobs=3, grid_resolution=20)
      plt.show()

  def plot_permutation_importances(self):
    # needs sklearn 0.22.
    self._check_fitted()
    v_df = self.v_df
    for j, param_name in enumerate(self.encoded_param_names):
      print('Importance plot for %s' % (param_name,))
      regressor = self.stat_estimators[param_name]
      imp1 = sklearn.inspection.permutation_importance(
          regressor, self.v_df, np_float(self.mech_params_stack[:, j]))
      sorted_idx = imp1.importances_mean.argsort()

      fig, ax = plt.subplots()
      ax.boxplot(
          imp1.importances[sorted_idx].T,
          vert=False,
          labels=v_df.columns[sorted_idx])
      ax.set_title('Permutation Importances (train set)')
      fig.tight_layout()
      plt.show()

  def predict(self, time_steps, num_samples, seed=0):
    self._check_fitted()
    rng = jax.random.PRNGKey(seed)
    mech_params = self.mech_params_stack
    # Mirrors the code in StatMechEstimator.predict from here on.
    predictions = monte_carlo.trajectories_from_model(self.mech_model,
                                                      mech_params, rng,
                                                      self.epidemics,
                                                      time_steps, num_samples)

    # TODO(jamieas): consider indexing by seed.
    sample = np.arange(num_samples)

    # Here we assume evenly spaced integer time values.
    # TODO(jamies): assert something during fit.
    epidemic_time = self.data.time.data
    time_delta = epidemic_time[1] - epidemic_time[0]
    time = np.arange(1, time_steps + 1) * time_delta + epidemic_time[-1]

    location = self.data.location

    return xarray.DataArray(
        predictions,
        coords=[location, sample, time],
        dims=['location', 'sample', 'time']).rename('new_infections')


def _get_static_covariate_df(trajectories):
  """The (static) covariate matrix."""
  raw_v_df = (
      trajectories.static_covariates.reset_coords(drop=True).transpose(
          'location', 'static_covariate').to_pandas())
  # This can then be used with, e.g. patsy.
  # expanded_v_df = patsy(raw_v_df, ...patsy details...)
  # Optionally it can be converted back to xa using.
  #  expanded_v_xa = xarray.DataArray(expanded_v_df)
  # for now...
  v_df = raw_v_df
  return v_df


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