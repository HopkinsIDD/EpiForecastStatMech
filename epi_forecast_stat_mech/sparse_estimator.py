# Lint as: python3
"""Implement the high_level interface for the sparse functionality."""

import collections
import pandas as pd
import pickle
import xarray

import numpy as np
import tensorflow as tf
import jax
from jax import numpy as jnp

from epi_forecast_stat_mech import high_level
from epi_forecast_stat_mech import sparse
from epi_forecast_stat_mech import viboud_chowell
from epi_forecast_stat_mech.evaluation import monte_carlo  # pylint: disable=g-bad-import-order
from epi_forecast_stat_mech.mechanistic_models import mechanistic_models  # pylint: disable=g-bad-import-order



def tf_float(x):
  return tf.convert_to_tensor(x, dtype=tf.float32)


def np_float(x):
  return np.asarray(x, dtype=np.float64)


def jnp_float(x):
  return jnp.asarray(x, dtype=jnp.float32)


def load(file_in):
  return pickle.load(file_in)


def _get_total_dof(result):
  return float(tf.reduce_sum(result.combo_result.soft_degrees_of_freedom))


def mech_model_class_from_intensity_family(intensity_family):
  model_name = intensity_family.name
  if model_name == 'Viboud-Chowell':
    return mechanistic_models.ViboudChowellModel
  elif model_name == 'Gaussian':
    return mechanistic_models.GaussianModel
  else:
    raise ValueError('%s is not implemented in mechanistic_models' % model_name)


class SparseEstimator(high_level.Estimator):

  def __init__(self,
               intensity_family=viboud_chowell.ViboudChowellFamily,
               initializer=sparse.common_fit_initializer,
               penalty_scale_init=None,
               mech_bottom_scale=None,
               penalty_factor_grid=tuple(
                   np.exp(np.linspace(np.log(.1), np.log(1000.), num=17)))):
    self.intensity_family = intensity_family
    self.initializer = initializer
    n_mech_params = len(intensity_family.encoded_param_names)
    if penalty_scale_init is None:
      penalty_scale_init = tf_float([1.] * n_mech_params)
    if mech_bottom_scale is None:
      mech_bottom_scale = intensity_family.mech_bottom_scale0
    else:
      mech_bottom_scale = tf_float(mech_bottom_scale)
    self.penalty_scale_init = penalty_scale_init
    self.mech_bottom_scale = mech_bottom_scale
    self.penalty_factor_grid = sorted(penalty_factor_grid)
    self.run_map = collections.OrderedDict()
    self.initialize_from_last_fit = True

  def run_one_bic(self, penalty_scale, verbosity=1):
    result = sparse.do_single_bic_run(
        intensity_family=self.intensity_family,
        trajectories=self.data,
        combo_params_init=self.combo_params_init,
        penalty_scale=penalty_scale,
        mech_bottom_scale=self.mech_bottom_scale,
        verbosity=verbosity,
        bic_multiplier=1.,
        fudge_scale=100.)
    self.run_map[tuple(np_float(penalty_scale))] = result
    return result

  def fit(self,
          data):
    self.data = data
    intensity_family = self.intensity_family
    initializer = self.initializer
    penalty_scale_init = self.penalty_scale_init
    mech_bottom_scale = self.mech_bottom_scale
    penalty_factor_grid = self.penalty_factor_grid

    # TODO(): consider using the top of the grid in the initializer instead.
    combo_params_init = initializer(
        intensity_family,
        data,
        penalty_scale_init,
        mech_bottom_scale)
    self.combo_params_init0 = self.combo_params_init = combo_params_init
    # This is here in case I want to return early for debugging reasons.
    # self.combo_params = combo_params_init

    log10_factor_weight = .4
    dof_extra_weight = 1.
    summary_accum = []
    self.result_accum = result_accum = []

    # Run in decreasing order of penalty_factor so that the
    # constant fit is an ok initializer and hopefully things can
    # improve incrementally and use warm starts.
    for run_index, penalty_factor in enumerate(penalty_factor_grid[::-1]):
      penalty_scale = penalty_factor * penalty_scale_init
      result = self.run_one_bic(penalty_scale)
      if self.initialize_from_last_fit:
        self.combo_params_init = result.combo_params
      result_accum.append(result)
      bic = float(result.combo_result.combined_bic)
      dof = _get_total_dof(result)
      log10_factor = np.log10(penalty_factor)
      bic_extra_pen1 = bic + log10_factor_weight * log10_factor
      bic_extra_pen2 = bic + log10_factor_weight * log10_factor - dof_extra_weight * dof
      opt_success = result.opt_status[0]
      summary_accum.append((run_index, penalty_factor, bic, dof, bic_extra_pen1,
                            bic_extra_pen2, opt_success))
    column_names = ('run_index', 'penalty_factor', 'bic', 'dof',
                    'bic_extra_pen1', 'bic_extra_pen2', 'opt_success')
    bic_df = pd.DataFrame(summary_accum, columns=column_names)
    self.bic_df = bic_df
    bic_selection = bic_df.iloc[bic_df['bic'].idxmax(), :]
    extra_pen_selection1 = bic_df.iloc[bic_df['bic_extra_pen1'].idxmax(), :]
    extra_pen_selection2 = bic_df.iloc[bic_df['bic_extra_pen2'].idxmax(), :]
    final_selection = extra_pen_selection1
    selections_df = pd.DataFrame(
        [
            final_selection, bic_selection, extra_pen_selection1,
            extra_pen_selection2
        ],
        index=pd.Index(['final_selection', 'bic', 'extra_pen1', 'extra_pen2']))
    self.selections_df = selections_df
    self.final_selection = final_selection
    result = result_accum[int(self.final_selection.run_index)]
    self.result = result
    self.combo_params = result.combo_params
    return self

  @property
  def mech_params_tf_stack(self):
    """Note: this is the raw params in the encoding of ef.viboud_chowell."""
    self._check_fitted()
    return tf.stack(self.combo_params.mech_params_raw, axis=0)

  @property
  def mech_params_df(self):
    self._check_fitted()
    assert self.intensity_family.name == 'Viboud-Chowell', 'only VC implemented'
    accum = []
    for mp in self.combo_params.mech_params_raw:
      wrapped_mp = self.intensity_family.params_wrapper().reset(mp)
      accum.append((
          float(wrapped_mp.r),
          float(wrapped_mp.a),
          float(wrapped_mp.p),
          float(wrapped_mp.K)))
    return pd.DataFrame(accum,
                        columns = ('r', 'a', 'p', 'K'),
                        index=self.data.location.to_index())

  @property
  def mech_params_for_jax_code(self):
    assert self.intensity_family.name == 'Viboud-Chowell', 'only VC implemented'
    mech_params_df = self.mech_params_df
    return jnp_float(np_float(
        tf.math.log(mech_params_df.values)))

  def summarize_fit_params(self):
    self._check_fitted()
    sparse.summarize_mech_param_fits(self.result, self.data)

  def print_mech_params(self):
    di_od = self.fitted_trajectories
    for loc, di in di_od.items():
      print(loc, di.fitted_params)

  @property
  def fitted_trajectories(self):
    self._check_fitted()
    di_list = sparse.make_demo_intensity_list(self.intensity_family, self.data)
    accum = collections.OrderedDict()
    for mp, di, loc_x in zip(self.combo_params.mech_params_raw, di_list, self.data.location):
      # TODO(): fix this str business.
      loc = str(loc_x.data)
      wrapped_mp = self.intensity_family.params_wrapper().reset(mp)
      di.set_fitted_params(wrapped_mp)
      accum[loc] = di
    return accum

  def diagnose_fitted_trajectories(self):
    di_od = self.fitted_trajectories
    ## TODO(): make nicer.
    for loc, di in di_od.items():
      print(loc)
      print(di.fitted_params)
      di.do_plot()

  @property
  def mech_model(self):
    return mech_model_class_from_intensity_family(self.intensity_family)()

  @property
  def epidemics(self):
    self._check_fitted()
    return high_level._pack_epidemics_record_tuple(self.data.fillna(0))

  def _check_fitted(self):
    if not hasattr(self, 'combo_params'):
      raise AttributeError('`fit` must be called before `predict`.')

  def save(self, file_out):
    pickle.dump(self, file_out)

  def predict(self, time_steps, num_samples, seed=0):
    self._check_fitted()
    rng = jax.random.PRNGKey(seed)
    mech_params = self.mech_params_for_jax_code
    # Mirrors the code in StatMechEstimator.predict from here on.
    # Note that epidemics and mech_model are properties.
    predictions = monte_carlo.trajectories_from_model(
        self.mech_model, mech_params, rng,
        self.epidemics, time_steps, num_samples)

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
