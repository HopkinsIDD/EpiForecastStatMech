"""Data Simulation using the VC model.

Write a function that generates simulated infection curves from VC model with a
single, static covariate.
"""

from . import data_model
import numpy as np
from scipy import stats
import xarray as xr

SPLIT_TIME=100


def final_size_poisson_dist(num_locations):
  """Return final size of the epidemic using a poisson distribution.

  Calculate final size of the epidemic using a single, randomly generated
  covariate for each location and sampling from a Poisson distribution.

  Args:
    num_locations: an int representing the number of locations to simulate

  Returns:
    final_sizes: a np.array of shape (num_locations, 1) consisting of the final
    size for each epidemic
    v: a np.array of shape (num_locations, 1) consisting of the randomly
    generated covariate for each epidemic
    alpha: a np.array of shape (1, 2) consisting of the weights for each
    covariate
  """
  v = np.random.uniform(0.0, 3.0, (num_locations, 1))
  alpha = np.array([[3, np.log10(3)]])
  final_sizes = 10**(alpha[0, 0] + alpha[0, 1] * v[:, 0])
  return final_sizes, v, alpha


def new_vc_simulation_model(num_samples,
                            num_locations,
                            num_time_steps,
                            num_static_covariates=1):
  """Return a zero data_model.new_model with extra simulation parameters.

  Args:
    num_samples: int representing the number of unique samples to run for each
      location
    num_locations: int representing the number of locations to model epidemics
      for
    num_time_steps: int representing the maximum number of time steps the
      epidemic can have
    num_static_covariates: int representing the number of static covariates for
      each location. Currently only 1 is supported.

  Returns:
    ds: an xr.Dataset representing the new infections and
      covariates for each location and representing the simulation parameters.
      All datavalues are initialized to 0.
  """
  if num_time_steps < SPLIT_TIME:
    raise ValueError('num_time_steps must be at least %d' % (SPLIT_TIME,))
  ds = data_model.new_model(num_samples, num_locations, num_time_steps,
                            num_static_covariates)
  ds['canonical_split_time'] = SPLIT_TIME
  ds['canonical_split_time'].attrs['description'] = (
      'Int representing the canonical time at which to split the data.')
  ds['growth_rate'] = data_model.new_dataarray({'location': num_locations})
  ds['growth_rate'].attrs['description'] = (
      'Float representing the growth rate in each location (r).'
      'This is used to simulate the number of new infections.')

  ds['growth_rate_exp'] = data_model.new_dataarray({'location': num_locations})
  ds['growth_rate_exp'].attrs['description'] = (
      'Float representing the extent that the growth rate is'
      'super or sub exponential (p).'
      'This is used to simulate the number of new infections.')

  return ds


def generate_ground_truth(pred_size, r, p, num_samples, num_time_steps):
  """Generate the epidemic curve observed to date using the VC model.

  Assume we start with 1 infected individual, and a disease progression
  based on a ViboudChowell model parameterized by r and p.
  Args:
    pred_size: a xr.DataArray representing the predicted total number of people
      to get the disease in each location
    r: a xr.DataArray representing the growth rate of the disease in each
      location
    p: a xr.DataArray representing the extent that the growth rate is super or
      sub exponential in each location
    num_samples: an int representing the number of samples to run at each
      location
    num_time_steps: an int representing the number of simulation 'days' to run
      at each location.

  Returns:
    new_infections: a xr.DataArray representing the new_infections at each
      (sample, location, time).
  """
  num_locations = r.sizes['location']

  new_infections = data_model.new_dataarray({
      'sample': num_samples,
      'location': num_locations,
      'time': num_time_steps
  }).astype(int)

  # start with 1 infected individual at time t=0
  total_infected = xr.ones_like(new_infections.sum('time'))
  frac_infected = total_infected.astype(float) / pred_size

  for t in range(num_time_steps):
    # Determine the number of new infections
    # by drawing from a poisson distribution
    # TODO(edklein) is this right?
    mu = total_infected**p * r * (1 - frac_infected)
    new_infections[dict(time=t)] = stats.poisson.rvs(xr.ufuncs.maximum(
        mu, 0.)).astype(int)

    # update for next round
    total_infected = new_infections.sum('time')
    frac_infected = total_infected.astype(float) / pred_size

  return new_infections


def generate_simulations(final_size_fn,
                         num_samples,
                         num_locations,
                         num_time_steps=500,
                         constant_r=1.0,
                         constant_p=0.6,
                         fraction_infected_limits=(0.05, 1.0)):
  """Generate many samples of VC curves.

  Generate many VC curves. Each sample contains num_locations.
  The locations may have different covariates, and thus different trajectories.
  However between samples the covariates are the same, so the only difference
  is statistical.

  Args:
    final_size_fn: a partialfunction to generate the predicted final size of the
      epidemic when passed the number of locations.
    num_samples: an int representing the number of simulations to run
    num_locations: an int representing the number of epidemics to run in each
      simulation
    num_time_steps: an int representing the number of simulation 'days' (default
      500)
    constant_r: a float representing the constant growth rate (default 1.0)
    constant_p: a float representing the constant degree to which the growth
      rate is sub or super exponential (default 0.6)
    fraction_infected_limits: A pair of floats in [0, 1] representing the limits
      on the fraction of the population that will be infected at SPLIT_TIME.

  Returns:
    trajectories: a xr.Dataset of the simulated infections over time
  """
  # generate growth rate for all simulations,
  # this is constant between simulations
  final_sizes, v, alpha = final_size_fn(num_locations)

  num_static_covariates = v.shape[1]

  trajectories = new_vc_simulation_model(num_samples, num_locations,
                                         num_time_steps, num_static_covariates)
  trajectories['final_size'] = xr.DataArray(final_sizes, dims='location')
  trajectories['static_weights'] = xr.DataArray(
      alpha, dims=('static_covariate', 'static_weight'))
  trajectories['static_covariates'] = xr.DataArray(
      v, dims=('location', 'static_covariate'))

  trajectories['growth_rate'].data = constant_r * np.ones(num_locations)
  trajectories['growth_rate_exp'].data = constant_p * np.ones(num_locations)

  # Randomly generate the fraction of infected people for each
  # sample and location.
  trajectories['fraction_infected'] = xr.DataArray(np.random.uniform(
      fraction_infected_limits[0], fraction_infected_limits[1],
      (num_samples, num_locations)), dims=['sample', 'location'])
  # Initially, all trajectories start at time 0.
  # The actual start_time will be updated to be consistent with
  # fraction_infected being infected at SPLIT_TIME.
  dummy_start_time = np.zeros((num_locations,), dtype=np.int32)
  trajectories['new_infections'] = generate_ground_truth(
      trajectories.final_size, trajectories.growth_rate,
      trajectories.growth_rate_exp, trajectories.sizes['sample'],
      trajectories.sizes['time'])

  cases = trajectories.new_infections.cumsum('time')

  target_cases = (trajectories['fraction_infected'] *
                  trajectories['final_size']).round()

  hit_times = cases.where(cases <= target_cases).argmax('time').data
  shifts = SPLIT_TIME - hit_times

  old_ni = trajectories.new_infections
  shifted_new_infections = xr.concat([
      xr.concat([
          old_ni.isel(sample=j, location=k).shift(
              time=shifts[j, k], fill_value=0)
          for k in range(old_ni.sizes['location'])], dim='location')
      for j in range(old_ni.sizes['sample'])], dim='sample')

  trajectories['new_infections'] = shifted_new_infections
  trajectories['start_time'] = xr.DataArray(shifts, dims=['sample', 'location'])

  return trajectories

