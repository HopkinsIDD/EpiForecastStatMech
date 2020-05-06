# -*- coding: utf-8 -*-
import collections
import itertools
import time

import xarray
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .intensity_family import IntensityFamily
from .tf_common import *
from .flatten_util import ravel_pytree
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


"""### Log-probs to model residuals at "plugin" scales."""

def gaussian_logprob_at_plugin_scale(x):
  # mean parameter is set at 0.
  # scale parameter is set at MLE: sqrt(mean(x ** 2))
  # If you evaluate the log prob of x under a Gaussian model with this mean
  # and scale and simplify, you arrive at the result.
  n = np.prod(get_shape(x))
  mse = tf.reduce_mean(x**2)
  return -(n / 2.) * (1. + np.log(2 * np.pi) + tf.math.log(mse))


def gaussian_logprob_with_bottom_scale(x, bottom_scale):
  # Morally, this is like gaussian_logprob_at_plugin_scale, but the MLE
  # for the scale parameter can go to 0, resulting in unbounded "utility"
  # of making residuals go to 0 -- often this is not sensible.
  # This function, therefore, becomes more linear at 0 at "scale" bottom_scale.
  n = np.prod(get_shape(x))
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
  n = tf_float(get_shape(x)[0])
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
        distribution.log_prob(tf_float(trajectory.new_infections)))


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
        distribution.log_prob(tf_float(trajectory.new_infections)))


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
            tf.sqrt(tf_float(tf_float(trajectory.new_infections)))))


#@@@ hack
# This one works pretty well so far.
if False:

  def get_mech_logprob(trajectory, intensity_model, intensity_params):
    y = tf.sqrt(tf_float(tf_float(trajectory.new_infections)))
    y_hat = tf.sqrt(intensity_model(trajectory, intensity_params))
    val = -tf.math.asinh(2. * tf.reduce_sum((y - y_hat)**2))
    return val


#@@@ hack
if True:

  def get_mech_logprob(trajectory, intensity_model, intensity_params):
    y = tf.sqrt(tf_float(trajectory.new_infections))
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

    return make_value_and_grad(mech_loss0)

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
      trajectories0_hat.time = np.arange(round(np.max(tf_float(trajectories0.time)) * 1.5) + 5)
    else:
      trajectories0_hat = trajectories0

    new_infections_hat = np_float(
        self.intensity_family.intensity(trajectories0_hat, self.fitted_params))

    if False:  # show non-log curves.
      plt.plot(tf_float(trajectories0.time), tf_float(trajectories0.new_infections),
               '-o')
      plt.plot(trajectories0_hat.time, new_infections_hat, '--+r')
      plt.show()

    plt.plot(tf_float(trajectories0.time), tf_float(trajectories0.new_infections), '-o')
    if len(trajectories0_hat.time) > 0:
      plt.plot(trajectories0_hat.time, new_infections_hat, '--+r')
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

def make_demo_intensity_list(intensity_family, trajectories, align_time_to_first=True):
  accum = []
  for location, nominal_trajectory in trajectories.groupby('location'):
    trajectory = nominal_trajectory.where(~nominal_trajectory.new_infections.isnull(), drop=True).copy()
    if align_time_to_first:
      try:
        first_case_ix = np.where(trajectory.new_infections > 0)[0][0]
      except IndexError:
        first_case_ix = len(trajectory.new_infections)
      integer_day = trajectory.time.values
      if first_case_ix < len(trajectory.new_infections):
        integer_day = integer_day - integer_day[first_case_ix]
      else:
        if first_case_ix >= 1:
          integer_day = integer_day - integer_day[first_case_ix - 1]
      trajectory['time'] = xarray.DataArray(integer_day, dims=('time',), coords=(integer_day,))
      trajectory = trajectory.isel(time=slice(first_case_ix, len(trajectory.new_infections)), drop=True)
    accum.append(
        DemoIntensityFamily(intensity_family).set_trajectory(
            trajectory))
  return accum


def find_common_fit(intensity_family, trajectories, use_nelder_mead=False):
  di_list = make_demo_intensity_list(intensity_family, trajectories)
  mech_logprobs = [di.get_mech_logprob() for di in di_list]
  neg_sum_mech_logprobs = lambda params: -tf.reduce_sum(
      tf.stack([lp(params) for lp in mech_logprobs]))

  unravel = lambda x: intensity_family.params_wrapper().reset(x)
  if use_nelder_mead:
    opt1 = wrap_minimize(
        neg_sum_mech_logprobs,
        intensity_family.params0._x,
        method='nelder-mead',
        options={'maxiter': 10000})
    common_fit_params = unravel(opt1.x)
    opt_status = (opt1.success, float(opt1.fun), opt1.nfev, opt1.message)
  else:
    val_and_grad_common_mech_loss = make_value_and_grad(neg_sum_mech_logprobs)
    common_fit_params, opt_status, opt1 = _lbfgs_optim(
      val_and_grad_common_mech_loss, intensity_family.params0._x, unravel)
  print(*opt_status)
  print(common_fit_params)
  return common_fit_params, di_list


def find_separate_fits(intensity_family, trajectories):
  di_list = make_demo_intensity_list(intensity_family, trajectories)
  for di in di_list:
    di.do_nelder_mead()
  return [di.fitted_params for di in di_list], di_list


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
        'mech_params_hat_stack', 'intensity_family', 'penalty_scale',
        'opt_status'
    ])):
  """combo_result: A ComboLogProbAndBIC combo_params: A ComboParams mech_params_stack: the axis=0 stack of mech_params_raw mech_params_hat_stack: the linear models fit for mech_params_stack. intensity_family: penalty_scale: @@ consider adding the rest of the do_single_bic_run arguments, e.g.

  trajectories, bic_multiplier, etc....
  """


def get_model_dims(intensity_family, trajectories):
  v_df = _get_static_covariate_df(trajectories)
  out_dim = len(intensity_family.encoded_param_names)
  in_dim = v_df.shape[1]
  n_trajectories = v_df.shape[0]
  return n_trajectories, in_dim, out_dim


def _lbfgs_optim(f, x0, unravel_func, max_iter=10000):
  opt1 = wrap_minimize(
      f,
      x0,
      jac=True,
      method='L-BFGS-B', # sometimes line-search failure.
      options={'maxiter': max_iter})
  opt_status = (opt1.success, float(opt1.fun), opt1.nfev, opt1.message)
  x = unravel_func(tf_float(opt1.x))
  return x, opt_status, opt1


def _powell_optim(f, x0, unravel_func, max_iter=10000):
  opt1 = wrap_minimize(
      f,
      x0,
      jac=True,
      method='powell', # slow.
      options={'maxiter': max_iter})
  opt_status = (opt1.success, float(opt1.fun), opt1.nfev, opt1.message)
  x = unravel_func(tf_float(opt1.x))
  return x, opt_status, opt1


def _tfp_lbfgs_optim(f, x0, unravel_func, max_iter=10000):
  opt1 = tfp.optimizer.lbfgs_minimize(
        f,
        x0,
        max_iterations=max_iter)
  opt_status = (not bool(opt1.converged.numpy()), float(opt1.objective_value), opt1.num_objective_evaluations, '')
  x = unravel_func(tf_float(opt1.position))
  return x, opt_status, opt1


def _adam_optim(f, x0, unravel_func, learning_rate=1E-2, max_iter=10000, verbose=1):
  opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
  x = tf.Variable(x0)
  for i in range(max_iter):
    loss, grad = f(x)
    opt.apply_gradients([(grad, x)])
    if verbose and i % 500 == 0:
      print(i, float(loss), float(tf.math.sqrt(tf.reduce_sum(grad ** 2))))
  opt_status = (True, float(loss), max_iter, '')
  opt1 = DummyData()
  opt1.x = x
  x = unravel_func(x)
  return x, opt_status, opt1


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


def do_single_bic_run(intensity_family,
                      trajectories,
                      combo_params_init,
                      penalty_scale,
                      mech_bottom_scale,
                      verbosity=1,
                      bic_multiplier=1.,
                      fudge_scale=100.,
                      optimizer=_lbfgs_optim):
  """Find a sparseness penalized alpha and bic score it.

  Note about the BIC calculations:
    The usual BIC formula is -2 * log_lik(theta_hat) + log(sample_size) * number_of_parameters.
    It is derived as using the Laplace approximation to the log of the predictive probability
    of the data (specifically, the log of the integral of the Likelihood against an improper
    flat prior) and and then multiplying by -2 (c.f. "deviance").
    
    Here, we *reject convention* and define the bic without the -2, so that it is an approximation
    of the log predictive probability, full stop. Accordingly, it is:
      log_lik(theta_hat) - log(sample_size) / 2 * number_of_parameters.
    Accordingly, big bic values are good here.
    A refinement would be to use the log_posterior in place of the log_lik. I.e. log_lik(theta) + log_prior(theta).
    And use the MAP for theta_hat. This is more or less what's done here. Except, not exactly. The
    LASSO inspired penalized log-likelihood is kind of like a log_posterior, but not exactly. This
    criterion is used to find the "MAP", when you view the penalty_scale as fixed and known. But when
    the bic criterion is computed we're currently only using the loglikelihood at this alpha_hat.
    
    Furthermore, how do we arrive at "k"? We do not use the full size of alpha. We are inspired by 
    "On the “degrees of freedom” of the lasso" https://projecteuclid.org/euclid.aos/1194461726
    Zou, Hastie, Tibshirani (see also the "IDEA" paper supplement https://www.embopress.org/doi/pdf/10.15252/msb.20199174).
    So the degrees of freedom is the number of non-zero elements of alpha.
  Definitions:
    out_dim = len(intensity_family.encoded_param_names)
    in_dim = len(t.v) for any t in trajectories.
  Arguments:
    intensity_family:
    trajectories:
    combo_params_init: A ComboParams.
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
    optimizer: _lbfgs_optim or something with the same signature.

  Returns:
    BICRunSummary
  """
  if not isinstance(combo_params_init, ComboParams):
    raise TypeError('combo_params_init is not a ComboParams')

  v_df = _get_static_covariate_df(trajectories)

  tf_v_orig = tf_float(v_df.values)
  tf_v_means = tf.reduce_mean(tf_v_orig, axis=0, keepdims=True)
  tf_v_sd = tf.reshape(tf_float(v_df.std(axis=0, ddof=1)), (1, -1))
  # The centered-scaled version simply called tf_v.
  # Internally, we'll use this as "X",
  # But in the final report we'll switch back to tf_v_orig as "X".
  tf_v = (tf_v_orig - tf_v_means) / tf_v_sd

  n_trajectories, in_dim, out_dim = get_model_dims(
      intensity_family, trajectories)
  stat_n = tf_float(n_trajectories)

  # combo_n_timepoints = tf_float(np.sum(
  #     [len(t.num_new_infections_over_time) for t in trajectories]))

  di_list = make_demo_intensity_list(intensity_family, trajectories)
  mech_logprobs = [di.get_mech_logprob() for di in di_list]

  try:
    tf.broadcast_to(mech_bottom_scale, (out_dim,))
  except:
    raise ValueError('mech_bottom_scale must broadcast to (out_dim,)')
  try:
    tf.broadcast_to(penalty_scale, (out_dim,))
  except:
    raise ValueError('penalty_scale must broadcast to (out_dim,)')
  # combo_params_init is in "public X" coordinates, so we must
  # revert the "final_..." adjustments.
  combo_params0 = ComboParams(
      intercept=combo_params_init.intercept + tf.squeeze(
          tf.matmul(tf_v_means / tf_v_sd,
                    combo_params_init.alpha * tf.reshape(tf_v_sd, (-1, 1))),
          axis=0),
      alpha=combo_params_init.alpha * tf.reshape(tf_v_sd, (-1, 1)),
      mech_params_raw=combo_params_init.mech_params_raw)

  combo_params0_flat, unravel_combo_params = ravel_pytree(combo_params0)

  def combo_logprob_and_bic(combo_params_flat):
    combo_params = unravel_combo_params(combo_params_flat)
    (intercept, alpha, mech_params_raw) = combo_params
    alpha_scaled = alpha  # / something about mech_params_stack?
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

  val_and_grad_combo_loss = make_value_and_grad(combo_loss)

  # Things to check if something goes wrong. :)
  # combo_params0
  # combo_params0_flat
  # unravel_combo_params(combo_params0_flat)
  # combo_logprob_and_bic(combo_params0_flat)
  # val_and_grad_combo_loss(combo_params0_flat)
  # unravel_combo_params(val_and_grad_combo_loss(combo_params0_flat)[1])

  combo_params, opt_status, opt1 = optimizer(
      val_and_grad_combo_loss, combo_params0_flat, unravel_combo_params)

  if verbosity >= 2 or (not opt_status[0] and verbosity >= 1):
    print('optimization status: %r' % (opt_status,))

  # This is the fitted linear model for "X" == tf_v.
  (intercept, alpha, mech_params_raw) = combo_params
  # To correct for centering, i.e. compute the linear model for "X" == tf_v_orig
  # We reason as follows:
  #  intercept + tf.matmul(tf_v, alpha) ==
  #  intercept + tf.matmul((tf_v_orig - tf_v_means) / tf_v_sd, alpha) ==
  #  (intercept - tf.squeeze(tf.matmul(tf_v_means / tf_v_sd, alpha), axis=0)) + tf.matmul(tf_v_orig, alpha / tf.reshape(tf_v_sd, (-1, 1)))
  # So defining:
  #   final_intercept = intercept - tf.squeeze(tf.matmul(tf_v_means / tf_v_sd, alpha), axis=0)
  #   final_alpha = alpha / tf.reshape(tf_v_sd, (-1, 1))
  # restores the linear model form for "X" == tf_v_orig.

  final_intercept = intercept - tf.squeeze(tf.matmul(tf_v_means / tf_v_sd, alpha), axis=0)
  final_alpha = alpha / tf.reshape(tf_v_sd, (-1, 1))
  final_combo_params = ComboParams(final_intercept, final_alpha, mech_params_raw)

  if verbosity >= 2:
    mech_params = [
        intensity_family.params_wrapper().reset(x) for x in mech_params_raw
    ]
    print(final_intercept, final_alpha)
    for mp in mech_params:
      print(mp)

  combo_result = combo_logprob_and_bic(tf_float(opt1.x))

  mech_params_stack = tf.stack(mech_params_raw)
  mech_params_hat_stack = intercept + tf.matmul(tf_v, alpha)

  # Sanity check:
  mech_params_hat_stack2 = final_intercept + tf.matmul(tf_v_orig, final_alpha)
  abs_err_check = np.max(np.abs(np_float(mech_params_hat_stack) - np_float(mech_params_hat_stack2)))
  assert abs_err_check < 1E-4, 'Standardization associated problem.'

  if verbosity >= 1:
    print(
        float(combo_result.combined_bic),
        float(tf.reduce_sum(combo_result.soft_degrees_of_freedom)),
        np_float(penalty_scale), np_float(combo_result.stat_bic),
        np_float(combo_result.mech_log_prob))

  return BICRunSummary(combo_result, final_combo_params, mech_params_stack,
                       mech_params_hat_stack, intensity_family, penalty_scale,
                       opt_status)

def _get_intercept_s(intercept, intensity_family):
  return pd.Series(intercept, index=pd.Index(
          intensity_family.encoded_param_names, name='encoded_param'))


def _get_alpha_df(alpha, v_df, intensity_family):
  alpha_df = pd.DataFrame(
      np_float(alpha),
      index=v_df.columns,
      columns=pd.Index(
          intensity_family.encoded_param_names, name='encoded_param'))
  return alpha_df


def summarize_mech_param_fits(run1, trajectories):
  v_df = _get_static_covariate_df(trajectories)
  intensity_family = run1.intensity_family
  (intercept, alpha, mech_params_raw) = run1.combo_params
  alpha_df = _get_alpha_df(alpha, v_df, intensity_family)
  mech_params_stack = run1.mech_params_stack
  mech_params_hat_stack = run1.mech_params_hat_stack

  for j, raw_param_name in enumerate(intensity_family.encoded_param_names):
    resid = np_float(mech_params_stack[:, j] - mech_params_hat_stack[:, j])
    corr_coef = np.corrcoef(
        np_float(mech_params_hat_stack[:, j]),
        np_float(mech_params_stack[:, j]))[0, 1]
    alpha_for_standarized_X = alpha_df.iloc[:, j] * v_df.std(axis=0)
    alpha_preso_df = pd.DataFrame(collections.OrderedDict([
        ('alpha', alpha_df.iloc[:, j]),
        ('alpha_for_standardized_X', alpha_for_standarized_X),
        ('abs_afs', np.abs(alpha_for_standarized_X))]))
    alpha_preso_df.sort_values('abs_afs', ascending=False, inplace=True)
    alpha_preso_df.drop(columns = 'abs_afs', inplace=True)
    print((
        '****************************************\n'
        'name={name}\n'
        'intercept={intercept},\n'
        'alpha=\n{alpha_preso_df},\n'
        'corr_coef={corr_coef}, resid_sd={resid_sd}, raw_param_sd={raw_param_sd}, hat_sd={hat_sd}'
    ).format(
        name=raw_param_name,
        intercept=intercept[j],
        alpha=alpha_df.iloc[:, j],
        alpha_for_standarized_X = alpha_df.iloc[:, j] * v_df.std(axis=0),
        alpha_preso_df = alpha_preso_df,
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


def combo_params_from_inits(mech_params_init,
                            model_dims,
                            alpha_init=None):
  """
    mech_params_init: Either a single instance of a mechanistic Model fit
      from the intensity_family, or castable to a list of them with an entry for
      every trajectory.
    model_dims:  (n_trajectories, in_dim, out_dim)
    alpha_init: None or valid value from previous run, i.e. tensor of
      (in_dim, out_dim)
   Returns:
     ComboParams.
   """
  n_trajectories, in_dim, out_dim = model_dims

  try:
    list(mech_params_init)
    mech_init_is_list = True
  except TypeError:
    mech_init_is_list = False
  if mech_init_is_list:
    mech_params0 = list(mech_params_init)
    if not len(mech_params0) == n_trajectories:
      raise ValueError('mech_params_init must be a single value '
                       'or a list of length n_trajectories.')
    intercept0 = tf.reduce_mean(tf.stack([
        mp._x for mp in mech_params0], axis=0), axis=0)
  else:
    mech_params0 = [mech_params_init._x] * n_trajectories
    intercept0 = mech_params_init._x
  assert get_shape(intercept0) == (out_dim,)
  if alpha_init is not None:
    raise NotImplementedError()
    # alpha0 = alpha_init
    # intercept0 = intercept0 - tf.squeeze(tf.matmul(tf_v_means / tf_v_sd, alpha0), axis=0)
  else:
    alpha0 = tf_float(np.zeros((in_dim, out_dim)))
  combo_params0 = ComboParams(intercept0, alpha0, mech_params0)
  return combo_params0


def predefined_constant_initializer(
    intensity_family,
    data,
    unused_penalty_scale,
    unused_mech_bottom_scale,
    **unused_kwargs):
  model_dims = get_model_dims(intensity_family, data)
  return combo_params_from_inits(intensity_family.params0, model_dims, alpha_init=None)


def common_fit_initializer(intensity_family,
                           data,
                           unused_penalty_scale,
                           unused_mech_bottom_scale,
                           use_nelder_mead=False,
                           **unused_kwargs):
  model_dims = get_model_dims(intensity_family, data)
  common_fit_params, _ = find_common_fit(
      intensity_family=intensity_family,
      trajectories=data,
      use_nelder_mead=use_nelder_mead)
  return combo_params_from_inits(common_fit_params, model_dims, alpha_init=None)


def powell_fit_initializer(
    intensity_family,
    data,
    penalty_scale,
    mech_bottom_scale,
    begin_with_common_fit=False,
    **kwargs):
  if begin_with_common_fit:
    combo_params_init1 = common_fit_initializer(
        intensity_family,
        data,
        penalty_scale,
        mech_bottom_scale,
        **kwargs)
  else:
    combo_params_init1 = predefined_constant_initializer(
      intensity_family,
      data,
      penalty_scale,
      mech_bottom_scale,
      **kwargs)
  result = do_single_bic_run(
      intensity_family,
      data,
      combo_params_init1,
      penalty_scale,
      mech_bottom_scale,
      optimizer=_powell_optim,
      **kwargs)
  combo_params_init2 = result.combo_params
  return combo_params_init2
