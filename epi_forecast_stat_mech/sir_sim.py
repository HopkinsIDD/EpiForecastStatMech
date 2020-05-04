"""Generate a single discrete time SIR model.
"""
from . import data_model
import numpy as np
from scipy import stats
import xarray as xr

# Generate Betas
# Beta, or the growth rate of the infection, depends on the covariates.
# Here we implement three different functional forms for the dependency.

SPLIT_TIME = 100


def generate_betas_from_single_random_covariate(num_locations):
  """Beta depend on a single covariate that is randomly generated.

  Args:
    num_locations: an int representing the number of locations to simulate

  Returns:
    beta: an xr.DataArray consisting of the growth rate
      for each epidemic
    v: an xr.DataArray consisting of the randomly generated covariate for each
      location
    alpha: an xr.DataArray consisting of the weights for each covariate
  """
  v = xr.DataArray(
      np.random.uniform(0.0, 1.0, (num_locations, 1)),
      dims=['location', 'static_covariate'])
  alpha = xr.DataArray(np.ones(1), dims=['static_covariate'])
  beta = 0.4 * np.exp(alpha @ v)

  return beta, v, alpha


def generate_betas_effect_mod(num_locations):
  """Betas depend on 2 discrete, randomly generated effects.

  Args:
    num_locations: an int representing the number of locations to simulate

  Returns:
    beta: an xr.DataArray consisting of the growth rate
      for each epidemic
    v: an xr.DataArray consisting of the randomly generated covariate for each
      location
    alpha: an xr.DataArray consisting of the weights for each covariate
  """
  v = xr.DataArray(np.random.binomial(1, 0.5, size=(num_locations, 2)),
                   dims={'location': num_locations, 'static_covariate': 2})
  hd = v.values[:, 0]
  ws = v.values[:, 1]
  beta_np = np.exp(np.log(1.5) + np.log(2.0) * (hd == 1) * (ws == 0))
  beta = xr.DataArray(beta_np, dims={'location': num_locations})

  return beta, v, xr.DataArray(np.array([1, 1]), dims={'static_covariate': 2})


def generate_betas_many_cov2(num_locations, num_pred, num_not_pred):
  """Betas depend on real valued vector of covariates.

  Args:
    num_locations: an int representing the number of locations to simulate
    num_pred: number of covariates that affect beta
    num_not_pred: number of covariates that do not affect beta

  Returns:
    beta: an xr.DataArray consisting of the growth rate
      for each epidemic
    v: an xr.DataArray consisting of the randomly generated covariate for each
      location
    alpha: an xr.DataArray consisting of the weights for each covariate
  """
  # generate random covariates
  # sample from range -1, 1 uniformly
  v = xr.DataArray(np.random.uniform(
      low=-1.0, high=1.0, size=(num_locations, num_pred + num_not_pred)),
                   dims={'location': num_locations,
                         'static_covariate': num_pred+num_not_pred})

  # construct weights for each covariate
  alpha_1 = np.ones(num_pred)
  alpha_0 = np.zeros(num_not_pred)
  alpha = xr.DataArray(np.concatenate((alpha_1, alpha_0), axis=0),
                       dims={'static_covariate': num_pred+num_not_pred})

  # this has a different functional form than we've seen before
  beta_np = 1 + np.exp(np.matmul(alpha.values, v.values.T))
  beta = xr.DataArray(beta_np, dims={'location': num_locations})

  return beta, v, alpha


def new_sir_simulation_model(num_samples, num_locations, num_time_steps,
                             num_static_covariates, num_dynamic_covariates=0):
  """Return a zero data_model.new_model with extra simulation parameters.

  Args:
    num_samples: int representing the number of unique samples to run for each
      location
    num_locations: int representing the number of locations to model epidemics
      for
    num_time_steps: int representing the maximum number of time steps the
      epidemic can have
    num_static_covariates: int representing the number of static covariates for
      each location
    num_dynamic_covariates: int representing the number of dynamic covariates
      for each location (defaults to 0).

  Returns:
    ds: an xr.Dataset representing the new infections and
      covariates for each location and representing the simulation parameters.
      All datavalues are initialized to 0.
  """
  if num_time_steps < SPLIT_TIME:
    raise ValueError('num_time_steps must be at least %d' % (SPLIT_TIME,))
  ds = data_model.new_model(num_samples, num_locations, num_time_steps,
                            num_static_covariates, num_dynamic_covariates)
  ds['canonical_split_time'] = SPLIT_TIME
  ds['canonical_split_time'].attrs['description'] = (
      'Int representing the canonical time at which to split the data.')

  ds['static_weights'] = data_model.new_dataarray(
      {'static_covariate': num_static_covariates})

  ds['dynamic_weights'] = data_model.new_dataarray(
      {'time': num_time_steps,
       'dynamic_covariate': num_dynamic_covariates})

  # TODO(edklein) should population_size be a covariate?
  ds['population_size'] = data_model.new_dataarray({'location': num_locations})
  ds['population_size'].attrs[
      'description'] = 'Int representing the population size in each location.'

  ds['fraction_infected'] = data_model.new_dataarray({
      'sample': num_samples,
      'location': num_locations
  })
  ds['fraction_infected'].attrs['description'] = (
      'Float representing the fraction of the population '
      'infected at the day %d.' % (SPLIT_TIME,))

  ds['start_time'] = data_model.new_dataarray({
      'sample': num_samples,
      'location': num_locations
  })
  ds['start_time'].attrs['description'] = (
      'Int representing the infection start time at each'
      'sample x location')

  ds['recovery_rate'] = data_model.new_dataarray({'location': num_locations})
  ds['recovery_rate'].attrs[
      'description'] = ('Float representing the recovery rate in each location.'
                        ' This is used in the SIR simulation of the epidemic.')

  if not num_dynamic_covariates:
    ds['growth_rate'] = data_model.new_dataarray({'location': num_locations})
    ds['growth_rate'].attrs[
        'description'] = ('Float representing the growth rate in each location.'
                          'This is used in the SIR simulation of the epidemic.')
  else:
    ds['growth_rate'] = data_model.new_dataarray({'location': num_locations,
                                                  'time': num_time_steps})
    ds['growth_rate'].attrs[
        'description'] = ('Float representing the growth rate in each location'
                          ' at each point in time.'
                          'This is used in the SIR simulation of the epidemic.')

  return ds


def generate_ground_truth(population_size,
                          infection_start_time,
                          beta,
                          gamma,
                          num_samples,
                          num_time_steps):
  """A function that generates infections over time using a discrete SIR model.

  We assume that the epidemic starts with a single case at time 0.
  We then simulate the number of infected individuals as a function of time,
  until the number of infected individuals is 0.
  This is the epidemic curve. Returns the epidemic curves as a function of time.

  Args:
    population_size: a xr.DataArray representing the population size in each
      location
    infection_start_time: a xr.DataArray representing the start time of the
      infection in each location.
    beta: a xr.DataArray representing the growth rate of the disease in each
      location
    gamma: a xr.DataArray representing the recovery rate of the disease in each
      location
    num_samples: an int representing the number of samples to run at each
      location
    num_time_steps: an int representing the number of simulation 'days' to run
      at each location.

  Returns:
    new_infections: a xr.DataArray representing the new_infections at each
      (sample, location, time).
  """
  num_locations = population_size.sizes['location']

  num_recovered = data_model.new_dataarray({
      'sample': num_samples,
      'location': num_locations,
  }).astype(int)

  new_infections = data_model.new_dataarray({
      'sample': num_samples,
      'location': num_locations,
      'time': num_time_steps
  }).astype(int)
  # at each start time, we have 1 infection
  new_infections[dict(time=infection_start_time)] = 1
  # setup for t-0
  num_infected = new_infections.sel(time=0).copy()

  num_susceptible = population_size.expand_dims({'sample': num_samples}).copy()
  num_susceptible -= num_infected

  beta_td = beta.expand_dims({'time': new_infections.sizes['time']})

  for t in range(0, new_infections.sizes['time']):
    # Calculate the probability that a person becomes infected
    # Python3 doesn't seem to work, so force a float

    frac_pop_infected = num_infected.astype(float) / population_size
    prob_infected = 1 - np.exp(-frac_pop_infected*beta_td[dict(time=t)])

    # Determine the number of new infections
    # By drawing from a binomial distribution
    # Record the number of infections that occured at this time point

    new_infections[dict(time=t)] = stats.binom.rvs(
        num_susceptible.astype(int), prob_infected)
    # Don't overwrite the first infection at the start time
    # TODO(edklein) this is a hack
    new_infections[dict(time=infection_start_time)] = 1

    # Calculate the probability that a person recovers
    prob_recover = 1 - np.exp(-gamma)

    # Determine the number of recoveries
    # by drawing from a binomial distribution
    num_new_recoveries = stats.binom.rvs(num_infected, prob_recover)

    # Update counts
    num_new_infections = new_infections[dict(time=t)]

    num_susceptible -= num_new_infections
    num_recovered += num_new_recoveries
    num_infected += num_new_infections - num_new_recoveries

  return new_infections


def generate_simulations(gen_beta_fn,
                         beta_gen_parameters,
                         num_samples,
                         num_locations,
                         num_time_steps=500,
                         constant_gamma=0.33,
                         constant_pop_size=10000,
                         fraction_infected_limits=(.05, 1.)):
  """Generate many samples of SIR curves.

  Generate many SIR curves. Each sample contains num_locations.
  The locations may have different covariates, and thus different trajectories.
  However between samples the covariates are the same,
  so the only difference is statistical.

  Args:
    gen_beta_fn: a function to generate the beta values for each epidemic
    beta_gen_parameters: a tuple containing all the parameters needed by
      gen_beta_fn
    num_samples: an int representing the number of samples to run
    num_locations: an int representing the number of locations to run in each
      sample
    num_time_steps: an int representing the number of simulation 'days'
      (default 500)
    constant_gamma: a float representing the constant recovery rate (default
      0.33)
    constant_pop_size: an int representing the constant population size (default
      10000)
    fraction_infected_limits: A pair of floats in [0, 1] representing the limits
      on the fraction of the population that will be infected at SPLIT_TIME.

  Returns:
    trajectories: a xr.Dataset of the simulated infections over time
  """
  # generate growth rate for all samples,
  # this is constant between samples
  beta, v, alpha = gen_beta_fn(*beta_gen_parameters)

  num_static_covariates = v.shape[1]

  trajectories = new_sir_simulation_model(num_samples, num_locations,
                                          num_time_steps, num_static_covariates,
                                          num_dynamic_covariates=1)

  trajectories['growth_rate'] = beta
  trajectories['static_weights'] = alpha
  trajectories['static_covariates'] = v

  trajectories['population_size'].data = constant_pop_size * np.ones(
      num_locations)
  trajectories['recovery_rate'].data = constant_gamma * np.ones(num_locations)

  # Randomly generate the fraction of infected people for each
  # sample and location.
  trajectories['fraction_infected'].data = np.random.uniform(
      fraction_infected_limits[0], fraction_infected_limits[1],
      (num_samples, num_locations))
  # Initially, all trajectories start at time 0.
  # The actual start_time will be updated to be consistent with
  # fraction_infected being infected at SPLIT_TIME.
  dummy_start_time = np.zeros((num_locations,), dtype=np.int32)
  trajectories['new_infections'] = generate_ground_truth(
      trajectories.population_size, dummy_start_time, trajectories.growth_rate,
      trajectories.recovery_rate, trajectories.sizes['sample'],
      trajectories.sizes['time'])
  cases = trajectories.new_infections.cumsum('time')
  trajectories['final_size'] = final_size = cases.isel(time=-1)
  target_cases = (trajectories['fraction_infected'] *
                  trajectories['final_size']).round()
  hit_times = np.apply_along_axis(
      lambda x: np.where(x)[0][0], axis=-1, arr=cases >= target_cases)
  shifts = SPLIT_TIME - hit_times

  old_ni = trajectories.new_infections
  shifted_new_infections = xr.concat([
      xr.concat([
          old_ni.isel(sample=j, location=k).shift(
              time=shifts[j, k], fill_value=0)
          for k in range(old_ni.sizes['location'])
      ],
                dim='location')
      for j in range(old_ni.sizes['sample'])
  ],
                                     dim='sample')
  trajectories['new_infections'] = shifted_new_infections
  trajectories['start_time'].data = shifts
  return trajectories
