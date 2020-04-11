# -*- coding: utf-8 -*-
import collections
import itertools
import time

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import scipy
from scipy import stats
import scipy.optimize as optimize
import matplotlib.pyplot as plt
import seaborn as sns

tfd = tfp.distributions
tfb = tfp.bijectors


from .intensity_family import IntensityFamily
from .tf_common import *
from . import soft_laplace


# Process mcmc output helpers

def make_chain_states_df(chain_states, features, feature_name):
  cs_df = pd.DataFrame(
      np_float(chain_states),
      columns=pd.Index(features, name=feature_name),
      index=pd.Index(np.arange(chain_states.shape[0]), name='t'))
  return cs_df


def tall_version(df):
  tall_df = df.stack().copy()
  tall_df.name = 'val'
  tall_df = pd.DataFrame(tall_df).reset_index()
  return tall_df

"""### ravel_pytree for tf"""

# Implement a tf-compatible ravel_pytree, similar to jax.flatten_util.ravel_pytree.
# Built off of jax's tree_util.
import functools
from jax import tree_util


def slice_and_reshape(offset, next_offset, new_shape, flat):
  # print(offset, next_offset, new_shape)
  return tf.reshape(flat[offset:next_offset], new_shape)


def make_flat_and_unravel_list(leaves):
  flat_accum = []
  unravel_accum = []
  offset = 0
  for leaf in leaves:
    flat_leaf = tf.reshape(leaf, (-1,))
    flat_accum.append(flat_leaf)
    leaf_shape = tuple(leaf.shape.as_list())
    next_offset = offset + flat_leaf.shape.as_list()[0]
    # print(offset, next_offset, leaf_shape)
    # This doesn't work because closures only use the named object in this namespace
    # In particular, offset and so on keep changing.
    # unravel_accum.append(lambda flat: tf.reshape(flat[offset:next_offset], leaf_shape))
    # This works.
    unravel_accum.append(
        functools.partial(slice_and_reshape, offset, next_offset, leaf_shape))
    offset = next_offset

  def unravel_list(flat):
    accum = []
    for unravel in unravel_accum:
      accum.append(unravel(flat))
    return accum

  return tf.concat(flat_accum, axis=0), unravel_list


def ravel_pytree(pytree):
  leaves, treedef = tree_util.tree_flatten(pytree)
  flat, unravel_list = make_flat_and_unravel_list(leaves)
  unravel_pytree = lambda flat: treedef.unflatten(unravel_list(flat))
  return flat, unravel_pytree

"""### Log-probs to model residuals at "plugin" scales."""

def gaussian_logprob_at_plugin_scale(x):
  # mean parameter is set at 0.
  # scale parameter is set at MLE: sqrt(mean(x ** 2))
  # If you evaluate the log prob of x under a Gaussian model with this mean
  # and scale and simplify, you arrive at the result.
  n = np.prod(x.shape.as_list())
  mse = tf.reduce_mean(x**2)
  return -(n / 2.) * (1. + np.log(2 * np.pi) + tf.math.log(mse))


def gaussian_logprob_with_bottom_scale(x, bottom_scale):
  # Morally, this is like gaussian_logprob_at_plugin_scale, but the MLE
  # for the scale parameter can go to 0, resulting in unbounded "utility"
  # of making residuals go to 0 -- often this is not sensible.
  # This function, therefore, becomes more linear at 0 at "scale" bottom_scale.
  n = np.prod(x.shape.as_list())
  bottom_var = bottom_scale**2
  mse = tf.reduce_mean(x**2)
  return -(n / 2.) * (1. + np.log(2 * np.pi) + tf.math.asinh(mse / bottom_var) -
                      tf.math.log(2. / bottom_var))


def gaussian_logprob_with_bottom_scale_along_axis0(x, bottom_scale):
  # Same but acts across axis=0 only.
  # bottom_scale should be broadcastable to the shape of x with axis=0 dropped.
  # Morally, this is like gaussian_logprob_at_plugin_scale, but the MLE
  # for the scale parameter can go to 0, resulting in unbounded "utility"
  # of making residuals go to 0 -- often this is not sensible.
  # This function, therefore, becomes more linear at 0 at "scale" bottom_scale.
  n = tf_float(x.shape.as_list()[0])
  bottom_var = bottom_scale**2
  mse = tf.reduce_mean(x**2, axis=0)
  return -(n / 2.) * (1. + np.log(2 * np.pi) + tf.math.asinh(mse / bottom_var) -
                      tf.math.log(2. / bottom_var))


"""### @@hack: Experiment with different functional forms for the mech log likelihood."""

# Mechanistic modelling helpers

# This is the vanilla one.
if False:

  def get_distribution(trajectory, intensity_model, intensity_params):
    return tfd.Poisson(rate=intensity_model(trajectory, intensity_params))

  def get_mech_logprob(trajectory, intensity_model, intensity_params):
    distribution = get_distribution(trajectory, intensity_model,
                                    intensity_params)
    return tf.reduce_sum(
        distribution.log_prob(trajectory.num_new_infections_over_time))


if False:

  def get_distribution(trajectory, intensity_model, intensity_params):
    #@@@ hack
    # return tfd.Poisson(rate=intensity_model(trajectory, intensity_params))
    return tfd.Normal(
        loc=intensity_model(trajectory, intensity_params), scale=.1)

  def get_mech_logprob(trajectory, intensity_model, intensity_params):
    distribution = get_distribution(trajectory, intensity_model,
                                    intensity_params)
    return tf.reduce_sum(
        distribution.log_prob(trajectory.num_new_infections_over_time))


#@@@ hack
if False:

  def get_distribution(trajectory, intensity_model, intensity_params):
    #return tfd.Normal(loc=tf.sqrt(intensity_model(trajectory, intensity_params)), scale=.5)
    return tfd.Normal(
        loc=tf.sqrt(intensity_model(trajectory, intensity_params)), scale=5.)

  def get_mech_logprob(trajectory, intensity_model, intensity_params):
    distribution = get_distribution(trajectory, intensity_model,
                                    intensity_params)
    return tf.reduce_mean(
        distribution.log_prob(
            tf.sqrt(tf_float(trajectory.num_new_infections_over_time))))


#@@@ hack
# This one works pretty well so far.
if False:

  def get_mech_logprob(trajectory, intensity_model, intensity_params):
    y = tf.sqrt(tf_float(trajectory.num_new_infections_over_time))
    y_hat = tf.sqrt(intensity_model(trajectory, intensity_params))
    val = -tf.math.asinh(2. * tf.reduce_sum((y - y_hat)**2))
    return val


#@@@ hack
if True:

  def get_mech_logprob(trajectory, intensity_model, intensity_params):
    y = tf.sqrt(tf_float(trajectory.num_new_infections_over_time))
    y_hat = tf.sqrt(intensity_model(trajectory, intensity_params))
    val = gaussian_logprob_with_bottom_scale(y - y_hat, 0.5)
    return val


"""### Wrap IntensityFamily in a helper class: DemoIntensityFamily"""

class DummyData(object):
  pass


class DemoIntensityFamily(object):

  def __init__(self, intensity_family):
    self.intensity_family = intensity_family
    self.fitted_params = None

  def get_mech_loss0(self):

    @tf.function
    def mech_loss0(params):
      wrapped_params = self.intensity_family.params_wrapper().reset(params)
      return -get_mech_logprob(self.trajectories0,
                               self.intensity_family.intensity, wrapped_params)

    return mech_loss0

  def get_val_and_grad_mech_loss0(self):
    mech_loss0 = self.get_mech_loss0()

    @tf.function
    def val_and_grad_mech_loss0(x):
      with tf.GradientTape() as tape:
        tape.watch(x)
        loss = mech_loss0(x)
      grad = tape.gradient(loss, x)
      return loss, grad

    return val_and_grad_mech_loss0

  def set_trajectory(self, trajectories0):
    self.trajectories0 = trajectories0
    return self

  def set_fitted_params(self, fitted_params):
    self.fitted_params = fitted_params
    return self

  def do_nelder_mead(self):
    mech_loss0 = self.get_mech_loss0()
    opt1 = wrap_minimize(
        mech_loss0,
        self.intensity_family.params0._x,
        method='nelder-mead',
        options={'maxiter': 10000})
    wrapped_params = self.intensity_family.params_wrapper().reset(opt1.x)
    self.fitted_params = wrapped_params
    self.opt1 = opt1
    print(opt1.success, float(opt1.fun), opt1.nfev, opt1.message)
    print(wrapped_params)
    return self

  def do_lbfgs(self):
    val_and_grad_mech_loss0 = self.get_val_and_grad_mech_loss0()
    print(val_and_grad_mech_loss0(self.intensity_family.params0._x))
    opt1 = wrap_minimize(
        val_and_grad_mech_loss0,
        self.intensity_family.params0._x,
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': 10000})
    wrapped_params = self.intensity_family.params_wrapper().reset(opt1.x)
    self.fitted_params = wrapped_params
    self.opt1 = opt1
    print(opt1.success, float(opt1.fun), opt1.nfev, opt1.message)
    print(wrapped_params)
    return self

  def do_plot(self):
    can_extrapolate = not (self.intensity_family.name in ['Viboud-Chowell'])
    trajectories0 = self.trajectories0
    if can_extrapolate:
      trajectories0_hat = DummyData()
      trajectories0_hat.t = np.arange(round(np.max(trajectories0.t) * 1.5) + 5)
    else:
      trajectories0_hat = trajectories0

    num_new_infections_over_time_hat = np_float(
        self.intensity_family.intensity(trajectories0_hat, self.fitted_params))

    if False:  # show non-log curves.
      plt.plot(trajectories0.t, trajectories0.num_new_infections_over_time,
               '-o')
      plt.plot(trajectories0_hat.t, num_new_infections_over_time_hat, '--+r')
      plt.show()

    plt.plot(trajectories0.t, trajectories0.num_new_infections_over_time, '-o')
    plt.plot(trajectories0_hat.t, num_new_infections_over_time_hat, '--+r')
    plt.yscale('log')
    plt.show()
    return self

  def get_mech_logprob(self):

    @tf.function
    def mech_logprob(params):
      wrapped_params = self.intensity_family.params_wrapper().reset(params)
      return get_mech_logprob(self.trajectories0,
                              self.intensity_family.intensity, wrapped_params)

    return mech_logprob

  def do_mcmc(self,
              step_size=.005,
              num_leapfrog_steps=4,
              num_results=50000,
              num_burnin_steps=10000,
              num_steps_between_results=100):
    trajectories0 = self.trajectories0
    if self.fitted_params is not None:
      params0 = self.fitted_params
    else:
      params0 = self.intensity_family.params0

    mech_logprob = self.get_mech_logprob()
    kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=mech_logprob,
        step_size=np.float64(step_size),
        num_leapfrog_steps=num_leapfrog_steps)

    def trace_fn(current_state, kernel_results):
      return kernel_results

    # Go! We use tf.function to compile the code into a graph, which will be better
    # optimized than pure eager execution. We time things to see how long it took.
    print('...sampling...')
    start_time = time.time()
    chain_states, krs = tf.function(
        autograph=False,
        func=lambda: tfp.mcmc.sample_chain(
            num_results=num_results,
            current_state=params0._x,
            kernel=kernel,
            num_burnin_steps=num_burnin_steps,
            num_steps_between_results=num_steps_between_results,
            trace_fn=trace_fn))()
    end_time = time.time()
    self.chain_states = chain_states  # shape: (num_results, param_dim)
    self.krs = krs
    final_params = self.intensity_family.params_wrapper().reset(
        chain_states[-1, :])
    self.fitted_params = final_params
    # diagnostics
    print('Total time compiling + sampling', (end_time - start_time))
    print('Acceptance rate:', krs.is_accepted.numpy().mean())
    return self

  def do_mcmc_plots(self):
    print(self.chain_states.shape)
    cs_df = make_chain_states_df(self.chain_states,
                                 self.intensity_family.encoded_param_names,
                                 'raw_param')
    tall_cs_df = tall_version(cs_df)

    g = sns.FacetGrid(
        tall_cs_df, row='raw_param', sharey=False, height=2, aspect=4)
    g2 = g.map(plt.plot, 't', 'val')
    plt.show()

    sns.pairplot(cs_df.query('t >= 0'))
    plt.show()
    return self

"""## Fit utilities."""

def make_demo_intensity_list(intensity_family, trajectories):
  return [
      DemoIntensityFamily(intensity_family).set_trajectory(tr)
      for tr in trajectories
  ]

def find_common_fit(intensity_family, trajectories):
  di_list = make_demo_intensity_list(intensity_family, trajectories)
  mech_logprobs = [di.get_mech_logprob() for di in di_list]
  neg_sum_mech_logprobs = lambda params: -tf.reduce_sum(
      tf.stack([lp(params) for lp in mech_logprobs]))

  opt1 = wrap_minimize(
      neg_sum_mech_logprobs,
      intensity_family.params0._x,
      method='nelder-mead',
      options={'maxiter': 10000})

  common_fit_params = intensity_family.params_wrapper().reset(opt1.x)
  print(opt1.success, float(opt1.fun), opt1.nfev, opt1.message)
  print(common_fit_params)
  # for di in di_list:
  #   di.set_fitted_params(common_fit_params).do_plot()
  return common_fit_params, di_list

def soft_nonzero(x):
  sq = x**2
  return sq / (1. + sq)


"""### Stack IntensityFamily calls

### Draft a separate fitter with a statistical penalty term for all params with an alpha regularizer tracking degrees of freedom.
"""

class ComboParams(
    collections.namedtuple('ComboParams',
                           ['intercept', 'alpha', 'mech_params_raw'])):
  """mech_params_raw: a list of tf vectors suitable for the intensity_family.
  """


class ComboLogProbAndBIC(
    collections.namedtuple('ComboLogProbAndBIC', [
        'penalized_log_prob', 'combined_bic', 'stat_bic', 'mech_log_prob',
        'soft_degrees_of_freedom'
    ])):
  """
  """


class BICRunSummary(
    collections.namedtuple('BICRunSummary', [
        'combo_result', 'combo_params', 'mech_params_stack',
        'mech_params_hat_stack', 'intensity_family', 'penalty_scale'
    ])):
  """combo_result: A ComboLogProbAndBIC combo_params: A ComboParams mech_params_stack: the axis=0 stack of mech_params_raw mech_params_hat_stack: the linear models fit for mech_params_stack. intensity_family: penalty_scale: @@ consider adding the rest of the do_single_bic_run arguments, e.g.

  trajectories, bic_multiplier, etc....
  """


def do_single_bic_run(intensity_family,
                      trajectories,
                      single_mech_params_init,
                      penalty_scale,
                      mech_bottom_scale,
                      verbosity=1,
                      bic_multiplier=1.,
                      fudge_scale=100.):
  """@@

  Definitions:
    out_dim = len(intensity_family.encoded_param_names)
    in_dim = len(t.v) for any t in trajectories.
  Arguments:
    intensity_family:
    trajectories:
    penalty_scale: Morally, a vector of Lasso penalty "lambda" parameters. Must
      broadcast to shape (out_dim,).
    mech_bottom_scale: Conceptually a low value for a standard error for
      predicting each mechanistic parameter, below which, extra precision is not
      critical.
      Must broadcast to shape: (out_dim,)
    verbosity: (int) Prints some at >=1, lots at >= 2.
    bic_multiplier: (default 1.) optional overall bic scaling correction
    fudge_scale:  (default 100.) controls the extent of the quadratic part. see
      alpha_loss

  Returns:
    @@
  """
  trajectories_index = pd.Index([t.unique_id for t in trajectories],
                                name='unique_id')
  # the covariates.
  v_df = pd.DataFrame(
      np.stack([t.v for t in trajectories]), index=trajectories_index)
  tf_v_orig = tf_float(v_df.values)
  tf_v_means = tf.reduce_mean(tf_v_orig, axis=0, keepdims=True)
  # The centered version simply called tf_v. Internally, we'll use this as "X",
  # But in the final report we'll switch back to tf_v_orig as "X".
  tf_v = tf_v_orig - tf_v_means
  tf_v_col_sd = tf.reshape(tf_float(v_df.std(axis=0, ddof=1)), (-1, 1))

  out_dim = len(intensity_family.encoded_param_names)
  in_dim = v_df.shape[1]
  stat_n = tf_float(len(trajectories))

  # combo_n_timepoints = tf_float(np.sum(
  #     [len(t.num_new_infections_over_time) for t in trajectories]))

  di_list = make_demo_intensity_list(intensity_family, trajectories)
  mech_logprobs = [di.get_mech_logprob() for di in di_list]

  mech_params0 = [single_mech_params_init._x] * len(di_list)
  intercept0 = single_mech_params_init._x
  assert intercept0.shape.as_list() == [out_dim]
  alpha0 = tf_float(np.zeros((in_dim, out_dim)))
  combo_params0 = ComboParams(intercept0, alpha0, mech_params0)

  combo_params0_flat, unravel_combo_params = ravel_pytree(combo_params0)

  def combo_logprob_and_bic(combo_params_flat):
    combo_params = unravel_combo_params(combo_params_flat)
    (intercept, alpha, mech_params_raw) = combo_params
    alpha_scaled = tf_v_col_sd * alpha  # / something about mech_params_stack?
    # morally, alpha_loss is the absolute value of alpha_scaled,
    # but it's approx. quadratic from 0. to O(1. / fudge_scale.)
    alpha_loss = soft_laplace.e_half_loss(fudge_scale * alpha_scaled) / fudge_scale
    # shape: (out_dim,)
    scaled_alpha_loss_per_out = (
        tf.reduce_sum(alpha_loss, axis=0) * penalty_scale)
    # The "Degrees of Freedom of the Lasso" paper asserts that the number of
    # non-zero coefficients (not including intercept) is (an unbiased estimate of)
    # the degrees of freedom of the fitter.
    soft_degrees_of_freedom = tf.reduce_sum(
        soft_nonzero(fudge_scale * alpha_scaled), axis=0)  # shape: (out_dim,)
    mech_log_prob = tf.reduce_sum(
        tf.stack(
            [lp(params) for lp, params in zip(mech_logprobs, mech_params_raw)]))
    mech_params_stack = tf.stack(mech_params_raw)
    mech_params_hat_stack = intercept + tf.matmul(tf_v, alpha)
    # shape: (out_dim,)
    stat_log_prob = gaussian_logprob_with_bottom_scale_along_axis0(
        mech_params_stack - mech_params_hat_stack, mech_bottom_scale)
    # shape: (out_dim,)
    penalized_stat_log_prob = stat_log_prob - scaled_alpha_loss_per_out
    # scalar
    penalized_log_prob = tf.reduce_sum(penalized_stat_log_prob) + mech_log_prob
    # stat_bic (without the -2 scale, so big is good) -- i.e. just the base Laplace approx.
    # for the log predictive probability of (in this case) each column, in turn,
    # of the mech_params_stack data.
    # shape: (out_dim,)
    stat_bic = stat_log_prob - bic_multiplier / 2. * tf.math.log(
        stat_n) * soft_degrees_of_freedom
    combined_bic = tf.reduce_sum(stat_bic) + mech_log_prob
    return ComboLogProbAndBIC(penalized_log_prob, combined_bic, stat_bic,
                              mech_log_prob, soft_degrees_of_freedom)

  def combo_loss(combo_params_flat):
    combo_result = combo_logprob_and_bic(combo_params_flat)
    return -combo_result.penalized_log_prob

  @tf.function
  def val_and_grad_combo_loss(x):
    with tf.GradientTape() as tape:
      tape.watch(x)
      loss = combo_loss(x)
    grad = tape.gradient(loss, x)
    return loss, grad

  # Things to check if something goes wrong. :)
  # combo_params0
  # combo_params0_flat
  # unravel_combo_params(combo_params0_flat)
  # combo_logprob_and_bic(combo_params0_flat)
  # val_and_grad_combo_loss(combo_params0_flat)

  if True:
    opt1 = wrap_minimize(
        val_and_grad_combo_loss,
        combo_params0_flat,
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': 10000})
  else:  #SLOW! exceeds maxiter without converging.
    opt1 = wrap_minimize(
        combo_loss,
        combo_params0_flat,
        method='nelder-mead',
        options={'maxiter': 10000})

  combo_params = unravel_combo_params(tf_float(opt1.x))
  if verbosity >= 2:
    print(opt1.success, float(opt1.fun), opt1.nfev, opt1.message)

  # This is the fitted linear model for "X" == tf_v.
  (intercept, alpha, mech_params_raw) = combo_params
  # To correct for centering, i.e. compute the linear model for "X" == tf_v_orig
  # We reason as follows:
  #  intercept + tf.matmul(tf_v, alpha) ==
  #  intercept + tf.matmul(tf_v_orig - tf_v_means, alpha) == 
  #  (intercept - tf.squeeze(tf.matmul(tf_v_means, alpha), axis=0)) + tf.matmul(tf_v_orig, alpha)
  # So defining:
  #   final_intercept = intercept - tf.squeeze(tf.matmul(tf_v_means, alpha), axis=0)
  # restores the linear model form for "X" == tf_v_orig.
    
  final_intercept = intercept - tf.squeeze(tf.matmul(tf_v_means, alpha), axis=0)
  final_combo_params = ComboParams(final_intercept, alpha, mech_params_raw)

  if verbosity >= 2:
    mech_params = [
        intensity_family.params_wrapper().reset(x) for x in mech_params_raw
    ]
    print(final_intercept, alpha)
    for mp in mech_params:
      print(mp)

  combo_result = combo_logprob_and_bic(tf_float(opt1.x))

  mech_params_stack = tf.stack(mech_params_raw)
  mech_params_hat_stack = intercept + tf.matmul(tf_v, alpha)

  # Sanity check:
  mech_params_hat_stack2 = final_intercept + tf.matmul(tf_v_orig, alpha)
  abs_err_check = np.max(np.abs(np_float(mech_params_hat_stack) - np_float(mech_params_hat_stack2)))
  assert abs_err_check < 1E-6, 'Centering associated problem.'

  if verbosity >= 1:
    print(
        float(combo_result.combined_bic), np_float(penalty_scale),
        np_float(combo_result.stat_bic), np_float(combo_result.mech_log_prob))

  return BICRunSummary(combo_result, final_combo_params, mech_params_stack,
                       mech_params_hat_stack, intensity_family, penalty_scale)

def summarize_mech_param_fits(run1):
  intensity_family = run1.intensity_family
  (intercept, alpha, mech_params_raw) = run1.combo_params
  mech_params_stack = run1.mech_params_stack
  mech_params_hat_stack = run1.mech_params_hat_stack

  for j, raw_param_name in enumerate(intensity_family.encoded_param_names):
    resid = np_float(mech_params_stack[:, j] - mech_params_hat_stack[:, j])
    corr_coef = np.corrcoef(
        np_float(mech_params_hat_stack[:, j]),
        np_float(mech_params_stack[:, j]))[0, 1]
    print((
        '****************************************\n'
        'name={name}\n'
        'intercept={intercept}, alpha={alpha},\n'
        'corr_coef={corr_coef}, resid_sd={resid_sd}, raw_param_sd={raw_param_sd}, hat_sd={hat_sd}'
    ).format(
        name=raw_param_name,
        intercept=intercept[j],
        alpha=alpha[:, j],
        resid_sd=np.std(resid, ddof=1),
        hat_sd=np.std(np_float(mech_params_hat_stack[:, j]), ddof=1),
        raw_param_sd=np.std(np_float(mech_params_stack[:, j]), ddof=1),
        corr_coef=corr_coef))
    plt.plot(
        np_float(mech_params_hat_stack[:, j]),
        np_float(mech_params_stack[:, j]), 'o')
    plt.xlabel(raw_param_name + '_hat')
    plt.ylabel(raw_param_name)
    plt.axis('equal')
    plt.show()

