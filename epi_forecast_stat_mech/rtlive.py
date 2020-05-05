# Lint as: python3
"""A rolled out version of the rt.live model."""

from epi_forecast_stat_mech import data_model  # pylint: disable=g-bad-import-order
from epi_forecast_stat_mech import estimator_base  # pylint: disable=g-bad-import-order
import numpy as np
import pandas as pd
import scipy.stats as sps
import xarray as xr


class RtLiveEstimator(estimator_base.Estimator):
  """A rolled out version of the rt.live model, following github implementation.

  See https://github.com/k-sys/covid-19/blob/master/Realtime%20R0.ipynb.
  """

  def __init__(self, sigma=0.25, gamma=1 / 7):
    # An array of every allowed value of Rt.
    R_T_MAX = 12  # pylint: disable=invalid-name
    self.r_t_range = np.linspace(0, R_T_MAX, R_T_MAX * 100 + 1)

    # A matrix of P(Rt_today|Rt_yesterday), representing drift in Rt over time.
    self.sigma = sigma
    process_matrix = sps.norm(
        loc=self.r_t_range, scale=self.sigma).pdf(self.r_t_range[:, np.newaxis])
    process_matrix /= process_matrix.sum(axis=0)
    self.process_matrix = pd.DataFrame(
        process_matrix, index=self.r_t_range, columns=self.r_t_range)

    # Gamma is 1/serial interval, used for relating Rt to the expected number of
    # new cases each day. The value of 1/7 comes from somewhere in
    # https://wwwnc.cdc.gov/eid/article/26/7/20-0282_article
    # https://www.nejm.org/doi/full/10.1056/NEJMoa2001316
    self.GAMMA = gamma  # pylint: disable=invalid-name

    # Initial prior on Rt, before any measurements are taken into account.
    prior = sps.gamma(a=4).pdf(self.r_t_range)
    self.prior = prior / prior.sum()

    # Populated when fit() is called.
    self.posterior = None
    self.latest_k = None

  def _likelihood(self, sr: pd.DataFrame) -> xr.DataArray:
    """Returns P(sr|Rt) for given case counts sr."""

    lam = sr[:-1].values * np.exp(
        self.GAMMA * (self.r_t_range[:, np.newaxis, np.newaxis] - 1))
    poisson = sps.poisson(lam)
    likelihood = xr.DataArray(
        poisson.pmf(sr[1:].values),
        coords=(self.r_t_range, sr.index[1:], sr.columns),
        dims=('Rt', sr.index.name, sr.columns.name))
    total_likelihood = likelihood.sum(dim='Rt')
    if (total_likelihood == 0).any():
      # A literally unbelievable thing happened, probably because GAMMA is too
      # small.
      impossible_time_locs = np.argwhere(total_likelihood.values == 0.0)
      time, loc = impossible_time_locs[0]
      raise ValueError(
          f'Total likelihood is zero for {len(impossible_time_locs)} (time, '
          f'location) pairs. For example, all values of r_t have zero '
          f'probability for this change:\n{sr.iloc[time-1:time+1, loc]}\n'
          f'Likely solution: use a larger value of gamma to permit faster '
          f'decay of case counts, e.g. RtLiveEstimator(gamma=1.0).')
    return likelihood

  def _prepare_cases(self, cases: pd.DataFrame) -> pd.DataFrame:
    # Fill NaNs and 0's with 1's. This algorithm assumes the expected number of
    # cases today is a multiple of the number of cases yesterday, so if the
    # number of cases is ever 0 or missing, it's trouble.
    cases = cases.where(cases > 0, 1.0)

    window = cases.rolling(7, win_type='gaussian', min_periods=1, center=True)
    # Note: We have to round because poisson.pmf is zero for non-integers.
    return window.mean(std=2).round()

  def _get_posterior(self, sr: pd.DataFrame) -> pd.DataFrame:
    likelihoods = self._likelihood(sr)

    # Iteratively apply Bayes' rule.
    # The github version stores the posterior for each day, but we only care
    # about the final posterior for forecasting purposes.
    posterior = self.prior[:, np.newaxis]
    for current_day in sr.index[1:]:
      # Update for process noise.
      posterior = self.process_matrix.values @ posterior

      # Update based on measurement.
      numerator = likelihoods.sel(
          time=current_day).values * posterior  # P(k|R_t)P(R_t)
      denominator = np.sum(numerator, axis=0, keepdims=True)  # P(k)
      posterior = numerator / denominator

    return pd.DataFrame(posterior, index=self.r_t_range, columns=sr.columns)

  def fit(self, observations: xr.Dataset):
    data_model.validate_data(observations, require_no_samples=True)
    sr = observations['new_infections'].to_pandas().T
    sr = self._prepare_cases(sr)
    self.posterior = self._get_posterior(sr)
    self.latest_k = sr.iloc[-1]
    return self

  def predict(self, time_steps, num_samples, seed=0) -> xr.DataArray:
    np.random.seed(seed)
    ks = [np.stack([self.latest_k.values] * num_samples, axis=-1)]

    # Sample Rt from posterior.
    rts = []
    for loc in self.posterior.columns:
      rts.append(
          np.random.choice(
              self.r_t_range, p=self.posterior[loc], size=num_samples))
    rts = pd.DataFrame(rts, index=self.posterior.columns)

    for _ in range(time_steps):
      # Sample number of new cases today.
      lam = ks[-1] * np.exp(self.GAMMA * (rts - 1))
      ks.append(sps.poisson.rvs(lam))

      # Update Rt based on process noise.
      new_rts = []
      for rt in rts.values.ravel():
        new_rts.append(
            np.random.choice(self.r_t_range, p=self.process_matrix[rt]))
      rts = pd.DataFrame(np.array(new_rts).reshape(rts.shape), index=rts.index)

    if isinstance(self.latest_k.name, np.number):
      dates = [self.latest_k.name + i for i in range(time_steps + 1)]
    else:
      dates = [
          self.latest_k.name + pd.Timedelta(i, 'D')
          for i in range(time_steps + 1)
      ]
    return xr.DataArray(
        ks[1:],
        coords=(dates[1:], self.latest_k.index, range(num_samples)),
        dims=('time', self.latest_k.index.name, 'sample'))


def get_estimator_dict():
  estimator_dict = {}
  estimator_dict['rtlive'] = RtLiveEstimator()
  return estimator_dict
