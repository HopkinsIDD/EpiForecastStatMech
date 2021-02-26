# Lint as: python3
"""An iterative epi_forecast_stat_mech.estimator_base.Estimator."""
import collections
import functools
import pickle

from epi_forecast_stat_mech import data_model
from epi_forecast_stat_mech import estimator_base
from epi_forecast_stat_mech import mask_time
from epi_forecast_stat_mech import optim_lib
from epi_forecast_stat_mech.mechanistic_models import mechanistic_models
from epi_forecast_stat_mech.mechanistic_models import predict_lib
from epi_forecast_stat_mech.statistical_models import probability as stat_prob
import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
import numpy as np
import sklearn
import sklearn.inspection


def np_float(x):
  return np.asarray(x, dtype=np.float64)


def jnp_float(x):
  return jnp.asarray(x, dtype=jnp.float32)


def load(file_in):
  return pickle.load(file_in)


class IterativeEstimator(estimator_base.Estimator):

  def save(self, file_out):
    pickle.dump(self, file_out)

  def __init__(self,
               stat_estimators=None,
               mech_model=None,
               hat_interpolation_alpha=0.5,
               iter_max=100,
               gradient_steps=10000,
               learning_rate=1E-4,
               verbose=1,
               time_mask_fn=functools.partial(mask_time.make_mask, min_value=1),
               rng=None
               ):
    """Construct an IterativeEstimator.

    Args:
      stat_estimators: A dictionary with an sklearn regressor for each
        encoded_param_name of the mech_model. Using a defaultdict is a
        convenient way to impose a default type of regressor, e.g.:
        collections.defaultdict(lambda: sklearn.ensemble.RandomForestRegressor(
          ...)). The fitted estimators will be saved in this dictionary, which
        is also saved on the instance.
      mech_model: An instance of a MechanisticModel.
      hat_interpolation_alpha: float between 0. and 1. representing how much
        to move toward the newly estimated mech_params_hat in each loop.
      iter_max: positive integer. How many iterative loops to perform.
      gradient_steps: The number of adam steps to perform per iter.
      learning_rate: The learning rate (positive float; default 1E-4).
      verbose: (Integer >= 0, default 1) Verbosity:
        0: Quiet.
        1: Reports every 1k steps.
        2: Also report initial value and gradient.
      time_mask_fn: A function that returns a np.array that can be used to mask
        part of the new_infections curve.
      rng: A jax.random.PRNGKey.
    """
    if stat_estimators is None:
      stat_estimators = collections.defaultdict(
          lambda: sklearn.ensemble.RandomForestRegressor(n_estimators=50))
    self.stat_estimators = stat_estimators
    self.hat_interpolation_alpha = hat_interpolation_alpha
    self.iter_max = iter_max
    self.gradient_steps = gradient_steps
    self.learning_rate = learning_rate
    self.verbose = verbose
    if mech_model is None:
      mech_model = mechanistic_models.ViboudChowellModel()
    self.mech_model = mech_model
    self.encoded_param_names = self.mech_model.encoded_param_names
    self.mech_bottom_scale = self.mech_model.bottom_scale
    self.out_dim = len(self.encoded_param_names)
    self.time_mask_fn = time_mask_fn
    if rng is None:
      rng = jax.random.PRNGKey(0)
    self.rng = rng

  def _unflatten(self, x):
    return jnp.reshape(x, (-1, self.out_dim))

  def fit(self, data):
    data_model.validate_data_for_fit(data)
    self.data = data
    num_locations = data.sizes['location']
    self.epidemics = epidemics = mechanistic_models.pack_epidemics_record_tuple(
        data)
    self.time_mask = time_mask = self.time_mask_fn(data)
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
    mech_params_stack = mechanistic_models.initialize_mech_model_stack(
        self.rng, self.mech_model, data, epidemics)
    assert mech_params_stack.shape[1] == len(self.encoded_param_names)
    mech_params_hat_stack = mech_params_stack

    for _ in range(self.iter_max):
      # Update mech_params_stack "regularized" by current mech_params_hat_stack.
      # N.B. This is not a maximum likelihood update.
      # We run two optimizers consecutively to try to unstick each.
      adam_loop = optim_lib.get_adam_optim_loop(
          functools.partial(
              mech_plus_stat_loss_val_and_grad,
              mech_params_hat_stack=mech_params_hat_stack),
          learning_rate=self.learning_rate)
      mech_params_stack = adam_loop(
          mech_params_stack,
          train_steps=self.gradient_steps,
          verbose=self.verbose)
      mech_params_stack, opt_status, _ = optim_lib.lbfgs_optim(
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
    self._check_fitted()
    rng = jax.random.PRNGKey(seed)
    mech_params = self.mech_params_stack
    dynamic_covariates = predict_lib.prepare_dynamic_covariates(
        self.data, test_data)
    sample_mech_params_fn = getattr(
        self, 'sample_mech_params_fn', lambda rngkey, num_samples: jnp.swapaxes(
            jnp.broadcast_to(mech_params,
                             (num_samples,) + mech_params.shape), 1, 0))

    return predict_lib.simulate_dynamic_predictions(
        self.mech_model, mech_params, self.data, self.epidemics,
        dynamic_covariates, num_samples, rng, sample_mech_params_fn)

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

  @property
  def mech_params_for_jax_code(self):
    return self.encoded_mech_params.values


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


def make_mean_estimators():
  return collections.defaultdict(
      lambda: sklearn.dummy.DummyRegressor(strategy='mean'))


def get_estimator_dict():
  estimator_dict = {}
  estimator_dict['iterative_randomforest__VC'] = IterativeEstimator()
  estimator_dict['iterative_mean__VC'] = IterativeEstimator(
      stat_estimators=make_mean_estimators())
  estimator_dict['iterative_randomforest__VC_PL'] = IterativeEstimator(
      mech_model=mechanistic_models.ViboudChowellModelPseudoLikelihood())
  estimator_dict['iterative_mean__VC_PL'] = IterativeEstimator(
      stat_estimators=make_mean_estimators(),
      mech_model=mechanistic_models.ViboudChowellModelPseudoLikelihood())
  estimator_dict['iterative_randomforest__Gaussian_PL'] = IterativeEstimator(
      mech_model=mechanistic_models.GaussianModelPseudoLikelihood())
  estimator_dict['iterative_mean__Gaussian_PL'] = IterativeEstimator(
      stat_estimators=make_mean_estimators(),
      mech_model=mechanistic_models.GaussianModelPseudoLikelihood())
  return estimator_dict
