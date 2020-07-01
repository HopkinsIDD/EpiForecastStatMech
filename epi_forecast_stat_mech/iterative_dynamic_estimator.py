# Lint as: python3
"""An iterative epi_forecast_stat_mech.estimator_base.Estimator."""
import collections
import functools
import pickle

import jax
from jax import flatten_util
from jax.experimental import optimizers
import jax.numpy as jnp
from matplotlib import pyplot as plt
import numpy as np
import scipy
import sklearn
import sklearn.inspection
import xarray as xr

from epi_forecast_stat_mech import data_model
from epi_forecast_stat_mech import estimator_base
from epi_forecast_stat_mech.mechanistic_models import mechanistic_models
from epi_forecast_stat_mech.mechanistic_models import predict_lib  # pylint: disable=g-bad-import-order
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
  total = new_infections.cumsum('time', skipna=True)
  mask = np.asarray((total >= min_value) & (~new_infections.isnull())).astype(
      np.float32)
  return mask


class IterativeDynamicEstimator(estimator_base.Estimator):

  def save(self, file_out):
    pickle.dump(self, file_out)

  def __init__(self,
               stat_estimators=None,
               mech_model_class=None,
               hat_interpolation_alpha=0.5,
               iter_max=100,
               gradient_steps=10000,
               learning_rate=1E-4,
               fudge_scale=100.,
               alpha_loss_weight=1E-1,
               stat_loss_weight=1E0,
               verbose=1):
    """Construct an IterativeEstimator.

    Args:
      stat_estimators: A dictionary with an sklearn regressor for each
        encoded_param_name of the mech_model. Using a defaultdict is a
        convenient way to impose a default type of regressor, e.g.:
        collections.defaultdict(lambda: sklearn.ensemble.RandomForestRegressor(
          ...)). The fitted estimators will be saved in this dictionary, which
          is also saved on the instance.
      mech_model_class: A dynamic MechanisticModel class.
      hat_interpolation_alpha: float between 0. and 1. representing how much to
        move toward the newly estimated mech_params_hat in each loop.
      iter_max: positive integer. How many iterative loops to perform.
      gradient_steps: The number of adam steps to perform per iter.
      learning_rate: The learning rate (positive float; default 1E-4).
      fudge_scale: (positive float) (inversely) governs the scale at which
        alpha_loss curvature near 0 happens.
      alpha_loss_weight: (non-negative float) Governs the weight on the
        alpha_loss penalty, which concerns the grouped-lasso like penalty on
        dynamic_covariates' coefficients. Note that this doesn't
        affect the stat-model re-estimation step, only the mech_params to
        mech_params_hat re-estimation step.
      stat_loss_weight: (non-negative float) Governs the weight one the
        mech_params versus mech_params_hat error term. Note that this doesn't
        affect the stat-model re-estimation step, only the mech_params to
        mech_params_hat re-estimation step.
      verbose: (Integer >= 0, default 1) Verbosity:
        0: Quiet.
        1: Reports every 1k steps.
        2: Also report initial value and gradient.
    """
    if stat_estimators is None:
      stat_estimators = collections.defaultdict(
          lambda: sklearn.ensemble.RandomForestRegressor(n_estimators=50))
    self.stat_estimators = stat_estimators
    if mech_model_class is None:
      raise ValueError('mech_model_class is required.')
    self.mech_model_class = mech_model_class
    self.hat_interpolation_alpha = hat_interpolation_alpha
    self.iter_max = iter_max
    self.gradient_steps = gradient_steps
    self.learning_rate = learning_rate
    self.fudge_scale = fudge_scale
    self.alpha_loss_weight = alpha_loss_weight
    self.stat_loss_weight = stat_loss_weight
    self.verbose = verbose

  def center_dynamic_covariates(self, dynamic_covariates):
    return ((dynamic_covariates - self.dynamic_covariate_m) /
            self.dynamic_covariate_s)

  def _unflatten(self, x):
    return jnp.reshape(x, (-1, self.out_dim))

  def fit(self, data):
    data_model.validate_data_for_fit(data, require_dynamics=True)
    self.data = data.copy()
    self.dynamic_covariate_m = (
        data.dynamic_covariates.mean(('time', 'location')))
    self.dynamic_covariate_s = (
        data.dynamic_covariates.std(('time', 'location')) + 1E-4)
    # It's a bit ugly, but seems like the cleaner first pass. The dynamic
    # coefficents are actually defined on the centered data scale; to effect
    # this, we put centered data in place of the original here and effect the
    # same centering in the predict call.
    data['original_dynamic_covariates'] = data.dynamic_covariates
    data['dynamic_covariates'] = self.center_dynamic_covariates(
        data.dynamic_covariates)
    mech_model_class = self.mech_model_class
    # Theoretically there should be an instance per location
    # (in the mech_param_stack) or I should re-design the class to be
    # "stateless" i.e. this is a method, not an init.
    mech_model = mech_model_class(jax.random.PRNGKey(0), data.dynamic_covariate)
    self.mech_model = mech_model
    self.encoded_param_names = self.mech_model.encoded_param_names
    self.mech_bottom_scale = self.mech_model.bottom_scale
    self.out_dim = len(self.encoded_param_names)
    num_locations = data.sizes['location']
    self.epidemics = epidemics = mechanistic_models.pack_epidemics_record_tuple(
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
      # Morally, alpha_loss is the norm (across location) of
      # time-dependent-covariate regression coefficients (intercepts are not
      # penalized), but it's approx. quadratic from 0. to O(1. / fudge_scale.)
      alpha_scaled = (
          mech_params_stack *
          mech_model.flat_kernel_indicator[jnp.newaxis, :])
      alpha_loss = (
          soft_mean_square(self.fudge_scale * alpha_scaled, axis=0) /
          self.fudge_scale)
      penalized_mech_plus_current_stat_loss = (
          -(mech_log_prob + self.stat_loss_weight * jnp.sum(stat_log_prob)) +
          self.alpha_loss_weight * jnp.sum(alpha_loss))
      return penalized_mech_plus_current_stat_loss

    mech_plus_stat_loss_val_and_grad = jax.jit(
        jax.value_and_grad(mech_plus_stat_errors))

    mech_params_stack = jnp.stack(
        [self.mech_model.init_parameters() for _ in range(num_locations)])
    assert mech_params_stack.shape[1] == len(self.encoded_param_names)
    mech_params_hat_stack = mech_params_stack

    for _ in range(self.iter_max):
      # Update mech_params_stack "regularized" by current mech_params_hat_stack.
      # N.B. This is not a maximum likelihood update.
      # We run two optimizers consecutively to try to unstick each.
      adam_loop = get_adam_optim_loop(
          functools.partial(
              mech_plus_stat_loss_val_and_grad,
              mech_params_hat_stack=mech_params_hat_stack),
          learning_rate=self.learning_rate)
      mech_params_stack = adam_loop(
          mech_params_stack,
          train_steps=self.gradient_steps,
          verbose=self.verbose)
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

  def predict(self, test_data, num_samples, seed=0):
    dynamic_covariates = xr.concat([self.data.dynamic_covariates,
                                    test_data.dynamic_covariates], dim='time')
    centered_dynamic_covariates = self.center_dynamic_covariates(dynamic_covariates)
    # This API is subject to change in pending CLs.
    self._check_fitted()
    rng = jax.random.PRNGKey(seed)
    mech_params = self.mech_params_stack
    return predict_lib.simulate_dynamic_predictions(
        self.mech_model,
        mech_params,
        self.data,
        self.epidemics,
        centered_dynamic_covariates,
        num_samples,
        rng)

  @property
  def mech_params(self):
    self._check_fitted()
    return predict_lib.mech_params_array(self.data, self.mech_model,
                                         self.mech_params_stack)

  @property
  def mech_params_hat(self):
    self._check_fitted()
    return predict_lib.mech_params_array(self.data, self.mech_model,
                                         self.mech_params_hat_stack)

  @property
  def encoded_mech_params(self):
    self._check_fitted()
    return predict_lib.encoded_mech_params_array(self.data, self.mech_model,
                                                 self.mech_params_stack)


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


def get_adam_optim_loop(f, learning_rate=1E-3):
  opt_init, opt_update, get_params = optimizers.adam(learning_rate, eps=1E-6)

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
      opt_state, loss_value = repeated_train_step(step, fused_train_steps,
                                                  opt_state)
      if step % 1000 == 0:
        if verbose >= 1:
          print(f'Loss at step {step} is: {loss_value}.')

    x = get_params(opt_state)
    return x

  return train_loop


def soft_norm(x, axis):
  return jnp.sqrt(jnp.sum(jnp.square(x), axis=axis) + 1.) - 1.


def soft_mean_square(x, axis):
  return jnp.sqrt(jnp.mean(jnp.square(x), axis=axis) + 1.) - 1.


def make_mean_estimators():
  return collections.defaultdict(
      lambda: sklearn.dummy.DummyRegressor(strategy='mean'))


def get_estimator_dict():
  estimator_dict = {}
  estimator_dict[
      'iterative_mean__DynamicMultiplicative'] = IterativeDynamicEstimator(
          mech_model_class=mechanistic_models.DynamicMultiplicativeGrowthModel,
          stat_estimators=make_mean_estimators(), iter_max=20)
  estimator_dict[
      'iterative_mean__DynamicMultiplicative_reg_10'] = IterativeDynamicEstimator(
          mech_model_class=mechanistic_models.DynamicMultiplicativeGrowthModel,
          stat_estimators=make_mean_estimators(), iter_max=20,
          alpha_loss_weight=1E1,
          stat_loss_weight=1E1)
  estimator_dict[
      'iterative_mean__DynamicMultiplicative_reg_100'] = IterativeDynamicEstimator(
          mech_model_class=mechanistic_models.DynamicMultiplicativeGrowthModel,
          stat_estimators=make_mean_estimators(), iter_max=20,
          alpha_loss_weight=1E2,
          stat_loss_weight=1E2)
  estimator_dict['iterative_randomforest__DynamicMultiplicative'] = (
      IterativeDynamicEstimator(
          mech_model_class=mechanistic_models.DynamicMultiplicativeGrowthModel,
          stat_estimators=None,
          iter_max=20))
  estimator_dict[
      'iterative_mean__DynamicBaselineSEIRModel'] = IterativeDynamicEstimator(
          mech_model_class=mechanistic_models.DynamicBaselineSEIRModel,
          stat_estimators=make_mean_estimators(),
          iter_max=20)
  estimator_dict[
      'iterative_mean__DynamicBaselineSEIRModel_reg_10'] = IterativeDynamicEstimator(
          mech_model_class=mechanistic_models.DynamicBaselineSEIRModel,
          stat_estimators=make_mean_estimators(),
          iter_max=100, stat_loss_weight=1E1, alpha_loss_weight=1E-2)
  return estimator_dict
