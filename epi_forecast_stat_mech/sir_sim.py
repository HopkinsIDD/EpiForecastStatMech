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


def generate_betas_many_cov2(num_locations, num_pred=1, num_not_pred=2):
  """Betas depend on real valued vector of covariates.

  Args:
    num_locations: an int representing the number of locations to simulate.
    num_pred: an int representing the number of covariates that affect beta.
    num_not_pred: an int representing the number of covariates that do not
      affect beta.

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


def gen_dynamic_beta_random_time(num_locations, num_time_steps):
  """Betas change at a random time between 1 and num_time_steps-1.

  Args:
    num_locations: an int representing the number of locations to simulate
    num_time_steps: an int representing the number of time steps to simulate

  Returns:
    beta: an xr.DataArray consisting of the growth rate
      for each epidemic with dimensions (location, time)
    v: an xr.DataArray consisting of the randomly generated covariate for each
      location with dimensions (location, time, 1)
    alpha: an xr.DataArray consisting of the weights for each covariate with
      dimension 1.
  """
  time = np.random.randint(1, num_time_steps-1, num_locations)
  cov = np.zeros((num_locations, num_time_steps, 1))
  for i in range(num_locations):
    cov[i][time[i]:] = 1
  v = xr.DataArray(cov, dims=['location', 'time', 'dynamic_covariate'])
  alpha = np.random.uniform(-1., 0.)*xr.DataArray(np.ones(1), dims=['dynamic_covariate'])
  beta = 0.4 * np.exp(alpha @ v)
  return beta, v, alpha


def gen_social_distancing_weight(num_locations):
  alpha = np.random.uniform(-1., 0., 1)
  return alpha


def new_sir_simulation_model(num_locations, num_time_steps,
                             num_static_covariates, num_dynamic_covariates=0):
  """Return a zero data_model.new_model with extra simulation parameters.

  Args:
    num_locations: int representing the number of locations to model epidemics
      for
    num_time_steps: int representing the maximum number of time steps the
      epidemic can have
    num_static_covariates: int representing the number of static covariates for
      each location
    num_dynamic_covariates: int representing the number of dynamic covariates
      for each location.

  Returns:
    ds: an xr.Dataset representing the new infections and
      covariates for each location and representing the simulation parameters.
      All datavalues are initialized to 0.
  """
  if num_time_steps < SPLIT_TIME:
    raise ValueError('num_time_steps must be at least %d' % (SPLIT_TIME,))
  ds = data_model.new_model(num_locations, num_time_steps,
                            num_static_covariates, num_dynamic_covariates)
  ds['canonical_split_time'] = SPLIT_TIME
  ds['canonical_split_time'].attrs['description'] = (
      'Int representing the canonical time at which to split the data.')

  ds['static_weights'] = data_model.new_dataarray(
      {'static_covariate': num_static_covariates})

  # TODO(edklein) should population_size be a covariate?
  ds['population_size'] = data_model.new_dataarray({'location': num_locations})
  ds['population_size'].attrs[
      'description'] = 'Int representing the population size in each location.'

  ds['fraction_infected'] = data_model.new_dataarray({
      'location': num_locations
  })
  ds['fraction_infected'].attrs['description'] = (
      'Float representing the fraction of the population '
      'infected at the day %d.' % (SPLIT_TIME,))

  ds['start_time'] = data_model.new_dataarray({
      'location': num_locations
  })
  ds['start_time'].attrs['description'] = (
      'Int representing the infection start time at each location')

  ds['recovery_rate'] = data_model.new_dataarray({'location': num_locations})
  ds['recovery_rate'].attrs[
      'description'] = ('Float representing the recovery rate in each location.'
                        ' This is used in the SIR simulation of the epidemic.')

  if num_dynamic_covariates > 0:
    ds['dynamic_weights'] = data_model.new_dataarray(
        {'time': num_time_steps,
         'dynamic_covariate': num_dynamic_covariates})

    ds['growth_rate'] = data_model.new_dataarray({'location': num_locations,
                                                  'time': num_time_steps})
    ds['growth_rate'].attrs[
        'description'] = ('Float representing the growth rate in each location'
                          ' at each point in time.'
                          'This is used in the SIR simulation of the epidemic.')
  else:
    ds['growth_rate'] = data_model.new_dataarray({'location': num_locations})
    ds['growth_rate'].attrs[
        'description'] = ('Float representing the growth rate in each location.'
                          'This is used in the SIR simulation of the epidemic.')

  return ds


def _helper_ground_truth_setup(population_size,
                               num_time_steps):
  """A helper function that sets up time0 of an SIR simulation.

  This helper function calculates the number of susceptible, infected, and
  recovered individuals at the begining of a simulation. It returns these
  values to be used as initial values in _helper_ground_truth_loop.

  Args:
    population_size: a xr.DataArray representing the population size in each
      location
    num_time_steps: an int representing the number of simulation 'days' to run
      at each location.

  Returns:
    new_infections: a DataArray with shape (location, time).
      The infections at time 0 are initialized to 1 in all locations.
    num_susceptible: a DataArray with shape (location,) containing the
      number of susceptible individuals in each location at time 0.
    num_infected: a DataArray with shape (location,) containing the
      number of infected individuals (1) in each location at time 0.
    num_recovered: a DataArray with shape (location,) containing the
      number of recovered individuals in each location at time 0.
  """
  num_locations = population_size.sizes['location']

  num_recovered = data_model.new_dataarray({
      'location': num_locations,
  }).astype(int)

  new_infections = data_model.new_dataarray({
      'location': num_locations,
      'time': num_time_steps
  }).astype(int)
  # at each start time, we have 1 infection
  new_infections[dict(time=0)] = 1
  # setup for t-0
  num_infected = new_infections.sel(time=0).copy()

  num_susceptible = population_size.copy()
  num_susceptible -= num_infected
  return new_infections, num_susceptible, num_infected, num_recovered


def _helper_ground_truth_loop(num_susceptible, num_recovered, num_infected,
                              beta_time_t, gamma, population_size,
                              prob_infection_constant):
  """A helper function to calculate SIR for one time step of an SIR simulation.

  This helper function calculates the number of susceptible, infected, and
  recovered individuals at one time step. It returns these values to be used as
  initial values in the next call.

  Args:
    num_susceptible: a DataArray with shape (location,) containing the
      number of susceptible individuals in each location at time t.
    num_infected: a DataArray with shape (location,) containing the
      number of infected individuals (1) in each location at time t.
    num_recovered: a DataArray with shape (location,) containing the
      number of recovered individuals in each location at time t.
    beta_time_t: a xr.DataArray representing the growth rate of the disease in
      each location at time t.
    gamma: a xr.DataArray representing the recovery rate of the disease in each
      location
    population_size: a xr.DataArray representing the population size in each
      location
    prob_infection_constant: a float representing a constant that we multiply
      the probability of becoming infected by. We noticed that a value of 1. led
      to curves that were short in time and clustered in time. By changing this
      to less than 1., our models fit better.

  Returns:
    num_new_infections: a DataArray with shape (location,) containing the
      number of *new* infections that occured at thime t+1.
    num_susceptible: a DataArray with shape (location,) containing the
      number of susceptible individuals in each location at time t+1.
    num_infected: a DataArray with shape (location,) containing the
      number of infected individuals (1) in each location at time t+1.
    num_recovered: a DataArray with shape (location,) containing the
      number of recovered individuals in each location at time t+1.
  """
  # Calculate the probability that a person becomes infected
  # Python3 doesn't seem to work, so force a float

  frac_pop_infected = num_infected.astype(float) / population_size
  prob_infected = prob_infection_constant * (
      1 - np.exp(-frac_pop_infected * beta_time_t))
  # Make sure prob_infected is between 0 and 1
  prob_infected = prob_infected.where(prob_infected > 0, 0)
  prob_infected = prob_infected.where(prob_infected < 1, 1)

  # Determine the number of new infections
  # By drawing from a binomial distribution
  # Record the number of infections that occured at this time point
  num_new_infections = stats.binom.rvs(
      num_susceptible.astype(int), prob_infected)

  # Calculate the probability that a person recovers
  prob_recover = 1 - np.exp(-gamma)

  # Determine the number of recoveries
  # by drawing from a binomial distribution
  num_new_recoveries = stats.binom.rvs(num_infected, prob_recover)

  num_susceptible -= num_new_infections
  num_recovered += num_new_recoveries
  num_infected += num_new_infections - num_new_recoveries
  return num_new_infections, num_susceptible, num_recovered, num_infected


def generate_ground_truth(population_size,
                          beta,
                          gamma,
                          num_time_steps,
                          prob_infection_constant=0.2):
  """A function that generates infections over time using a discrete SIR model.

  We assume that the epidemic starts with a single case at time 0.
  We then simulate the number of infected individuals as a function of time,
  until the number of infected individuals is 0.
  This is the epidemic curve. Returns the epidemic curves as a function of time.

  Args:
    population_size: a xr.DataArray representing the population size in each
      location
    beta: a xr.DataArray representing the growth rate of the disease in each
      location
    gamma: a xr.DataArray representing the recovery rate of the disease in each
      location
    num_time_steps: an int representing the number of simulation 'days' to run
      at each location.
    prob_infection_constant: a float representing a constant that we multiply
      the probability of becoming infected by. We noticed that a value of 1. led
      to curves that were short in time and clustered in time. By changing this
      to less than 1., our models fit better.

  Returns:
    new_infections: a xr.DataArray representing the new_infections at each
      (location, time).
  """

  new_infections, num_susceptible, num_infected, num_recovered = _helper_ground_truth_setup(
      population_size, num_time_steps)

  if 'time' not in beta.dims:
    beta = beta.expand_dims({'time': new_infections.sizes['time']})

  for t in new_infections.time[1:]:
    beta_time_t = beta[dict(time=t)]
    num_new_infections, num_susceptible, num_recovered, num_infected = _helper_ground_truth_loop(
        num_susceptible, num_recovered, num_infected, beta_time_t, gamma,
        population_size, prob_infection_constant)
    new_infections[dict(time=t)] = num_new_infections

  return new_infections


def generate_social_distancing_ground_truth(population_size,
                                            beta,
                                            gamma,
                                            num_time_steps,
                                            social_distancing_threshold,
                                            gen_social_distancing_weight_fn,
                                            prob_infection_constant=0.2
                                            ):
  """Generate infections over time using SIR with a variable growth rate.

  We assume that the epidemic starts with a single case at time 0.
  We then simulate the number of infected individuals as a function of time.
  When the number of infected individuals reaches num_infected_threshold,
  we decrease the growth rate by an amount determined by social_distance_fn.
  We continue simulating the number of infected individuals until we reach
  num_time_steps. This is the epidemic curve. Returns the epidemic curves as a
  function of time.

  Args:
    population_size: a xr.DataArray representing the population size in each
      location
    beta: a xr.DataArray representing the static growth rate of the disease in
      each location
    gamma: a xr.DataArray representing the recovery rate of the disease in each
      location
    num_time_steps: an int representing the number of simulation 'days' to run
      at each location.
    social_distancing_threshold: an array of ints of shape (num_locations)
      indicating the number of infections at each location when we change the
      growth rate.
    gen_social_distancing_weight_fn: A (partial) function that generates the
      weights of the social distancing covariate. Function is called with the
      argument num_locations.
    prob_infection_constant: a float representing a constant that we multiply
      the probability of becoming infected by. We noticed that a value of 1. led
      to curves that were short in time and clustered in time. By changing this
      to less than 1., our models fit better.

  Returns:
    beta_td: a xr.DataArray representing the time-dependent growth rate at each
      (location, time).
    dynamic_covariate: a xr.DataArray representing the time-dependent covariate
      at each (location, time). Currently fixed to be one covariate with
      a value of either 0 or 1.
    dynamic_weights: a xr.DataArray representing the weight of dynamic_covariate
      currently a 1d array with dimension ['dynamic_covariate'].
    new_infections: a xr.DataArray representing the new_infections at each
      (location, time).
  """

  new_infections, num_susceptible, num_infected, num_recovered = _helper_ground_truth_setup(
      population_size, num_time_steps)
  num_locations = population_size.sizes['location']

  # need to compute the change in growth rate at a given
  # infection load. This will be represented by a time-dependent covariate
  # that will be 0 or 1 in all locations. (It's possible we'll
  # want to allow the value of it to change eventually.) The weight will be
  # constant in time, although we might store it as time-dependent for
  # consistency/scalability
  dynamic_alpha = gen_social_distancing_weight_fn(num_locations)
  dynamic_weights = xr.DataArray(dynamic_alpha, dims=['dynamic_covariate'])

  beta_td = beta.copy()

  if 'time' not in beta_td.dims:
    beta_td = beta_td.expand_dims({'time': new_infections.sizes['time']}).copy()

  dynamic_covariate = xr.zeros_like(new_infections)
  dynamic_covariate = dynamic_covariate.expand_dims({'dynamic_covariate':1}).copy()

  # No locations start off above their threshold
  # at t=0
  infection_threshold = xr.zeros_like(beta).expand_dims({'dynamic_covariate':1})

  for t in new_infections.time[1:]:
    # Update growth rate if needed
    dynamic_covariate[dict(time=t)] = infection_threshold.astype(int)
    beta_td[dict(time=t)] = beta + (dynamic_weights @ infection_threshold.astype(int))

    beta_time_t = beta_td[dict(time=t)]
    num_new_infections, num_susceptible, num_recovered, num_infected = _helper_ground_truth_loop(
        num_susceptible, num_recovered, num_infected, beta_time_t, gamma,
        population_size, prob_infection_constant)
    new_infections[dict(time=t)] = num_new_infections

    # Check if we need to update growth rate
    total_infected = population_size - num_susceptible
    infection_threshold = (total_infected > social_distancing_threshold).T.expand_dims({'dynamic_covariate':1})

  return beta_td, dynamic_covariate, dynamic_weights, new_infections


def _helper_setup_sir_sim(gen_constant_beta_fn,
                          num_locations,
                          num_time_steps=500,
                          constant_gamma=0.33,
                          population_size=10000,
                          gen_dynamic_beta_fn=None):
  """Helper function to set up and store a bunch of variables in a xr.DataSet.

  Returns a xr.Dataset containing the growth rate, covariates, weights,
  population size, and recovery rate.

  Args:
    gen_constant_beta_fn: a partial function to generate the constant beta
      values for each epidemic when passed num_locations.
    num_locations: an int representing the number of locations to run
    num_time_steps: an int representing the number of simulation 'days'
      (default 500)
    constant_gamma: a float representing the constant recovery rate (default
      0.33)
    population_size: a xr.DataArray representing the population size in each
      location. If none, defaults to 10000 in each location.
    gen_dynamic_beta_fn: A function to generate the dynamic beta
      values for each epidemic when passed num_locations and num_time_steps.
      None if the betas are all static.

  Returns:
    trajectories: a xr.Dataset containing the growth rate, covariates, weights,
      population size, and recovery rate.
  """
  # generate growth rate for all locations
  beta, v, alpha = gen_constant_beta_fn(num_locations)

  if type(population_size) is int:
    population_size = xr.DataArray(population_size * np.ones(num_locations), dims=['location'])

  static_covariates = xr.concat((v, population_size), 'static_covariate')
  num_static_covariates = static_covariates.sizes['static_covariate']

  # give population_size a weight of 0
  static_weights = xr.concat((alpha, xr.DataArray(np.array([0]), dims=['static_covariate'])), 'static_covariate')

  if gen_dynamic_beta_fn:
    beta_td, v_td, alpha_td = gen_dynamic_beta_fn(num_locations, num_time_steps)
    num_dynamic_covariates = (v_td.dynamic_covariate)
    beta = beta_td + beta.expand_dims({'time': num_time_steps})

  else:
    num_dynamic_covariates=0

  trajectories = new_sir_simulation_model(num_locations,
                                          num_time_steps, num_static_covariates,
                                          num_dynamic_covariates)

  trajectories['growth_rate'] = beta
  trajectories['static_covariates'] = static_covariates
  trajectories['static_weights'] = static_weights

  if gen_dynamic_beta_fn:
    trajectories['dynamic_weights'] = alpha_td
    trajectories['dynamic_covariates'] = v_td

  trajectories['population_size'] = population_size
  trajectories['recovery_rate'].data = constant_gamma * np.ones(num_locations)

  return trajectories


def generate_simulations(gen_constant_beta_fn,
                         num_locations,
                         num_time_steps=500,
                         constant_gamma=0.33,
                         population_size=10000,
                         gen_dynamic_beta_fn=None,
                         fraction_infected_limits=(.05, 1.),
                         prob_infection_constant=0.2):
  """Generate many SIR curves.

  Generate infection curves for num_locations. The locations may have different
  covariates, and thus different trajectories.

  Args:
    gen_constant_beta_fn: a partial function to generate the constant beta
      values for each epidemic when passed num_locations.
    num_locations: an int representing the number of locations to run
    num_time_steps: an int representing the number of simulation 'days'
      (default 500)
    constant_gamma: a float representing the constant recovery rate (default
      0.33)
    population_size: a xr.DataArray representing the population size in each
      location. If none, defaults to 10000 in each location.
    gen_dynamic_beta_fn: A function to generate the dynamic beta
      values for each epidemic when passed num_locations and num_time_steps.
      None if the betas are all static.
    fraction_infected_limits: A pair of floats in [0, 1] representing the limits
      on the fraction of the population that will be infected at SPLIT_TIME.
    prob_infection_constant: a float representing a constant that we multiply
      the probability of becoming infected by. We noticed that a value of 1. led
      to curves that were short in time and clustered in time. By changing this
      to less than 1., our models fit better.

  Returns:
    trajectories: a xr.Dataset of the simulated infections over time
  """
  trajectories = _helper_setup_sir_sim(gen_constant_beta_fn,
                                       num_locations,
                                       num_time_steps,
                                       constant_gamma,
                                       population_size,
                                       gen_dynamic_beta_fn)

  # Initially, all trajectories start at time 0.
  # The actual start_time will be updated to be consistent with
  # fraction_infected being infected at SPLIT_TIME.
  trajectories['new_infections'] = generate_ground_truth(
      trajectories.population_size, trajectories.growth_rate,
      trajectories.recovery_rate,
      trajectories.sizes['time'], prob_infection_constant)

  return data_model.shift_timeseries(trajectories, fraction_infected_limits, SPLIT_TIME)


def generate_social_distancing_simulations(gen_constant_beta_fn,
                                           gen_social_distancing_weight_fn,
                                           num_locations,
                                           num_time_steps=500,
                                           constant_gamma=0.33,
                                           population_size=10000,
                                           social_distancing_threshold=10000/4,
                                           fraction_infected_limits=(.05, 1.),
                                           prob_infection_constant=0.2,
                                           shift_timeseries=True):
  """Generate many SIR curves with social distancing.

  Generate many SIR curves with social distancing implemented when the number of
  cumulative infections reaches fraction_infected_limits. The locations may have
  different covariates, and thus different trajectories.

  Args:
    gen_constant_beta_fn: a partial function to generate the constant beta
      values for each epidemic when passed num_locations.
    gen_social_distancing_weight_fn: A (partial) function that generates the
      weights of the social distancing covariate. Function is called with the
      argument num_locations.
    num_locations: an int representing the number of locations to run
    num_time_steps: an int representing the number of simulation 'days'
      (default 500)
    constant_gamma: a float representing the constant recovery rate (default
      0.33)
    population_size: a xr.DataArray representing the population size in each
      location. If none, defaults to 10000 in each location.
    social_distancing_threshold: a DataArray representing the number of
      infected individuals in each location when we implement social distancing.
    fraction_infected_limits: A pair of floats in [0, 1] representing the limits
      on the fraction of the population that will be infected at SPLIT_TIME.
    prob_infection_constant: a float representing a constant that we multiply
      the probability of becoming infected by. We noticed that a value of 1. led
      to curves that were short in time and clustered in time. By changing this
      to less than 1., our models fit better.
    shift_timeseries: A bool indicating whether we should shift the trajectories
      based on fraction_infected_limits. If False, all trajectories will start
      with 1 infection at time t=0.

  Returns:
    trajectories: a xr.Dataset of the simulated infections over time
  """
  # generate growth rate for all locations
  trajectories = _helper_setup_sir_sim(gen_constant_beta_fn,
                                       num_locations,
                                       num_time_steps,
                                       constant_gamma,
                                       population_size,
                                       gen_dynamic_beta_fn=None)

  beta, dynamic_covariates, dynamic_weights, new_infections = generate_social_distancing_ground_truth(
      trajectories.population_size, trajectories.growth_rate,
      trajectories.recovery_rate,
      trajectories.sizes['time'], social_distancing_threshold,
      gen_social_distancing_weight_fn, prob_infection_constant)

  trajectories['growth_rate'] = beta
  trajectories['dynamic_covariates'] = dynamic_covariates
  trajectories['dynamic_weights'] = dynamic_weights
  trajectories['new_infections'] = new_infections

  if not shift_timeseries:
    return trajectories
  else:
    return data_model.shift_timeseries(trajectories, fraction_infected_limits,
                                       SPLIT_TIME)
