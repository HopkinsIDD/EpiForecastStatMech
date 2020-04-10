"""# Data Simulation

# VC Model

Write a function that generates simulated infection curves from VC model with a single, static covariate.
"""

import numpy as np
from scipy import stats
import pandas as pd
import collections
import matplotlib.pyplot as plt

#TODO: Duplicates some code in sir_sim. Move to a higher-level module
meta_data = collections.namedtuple('meta_data', 
                                   ['num_epidemics', 
                                    'epidemic_id', 
                                    'pred_final_epidemic_size', 
                                    'percent_infected',  
                                    'r',
                                    'p'])
"""A named tuple to hold a single disease metadata.

num_epidemics: an int representing the total number of epidemics to simulate
epidemic_id: an int representing the identifier of this epidemic
pred_final_epidemic_size: an int representing the predicted final size of the epidemic
percent_infected: a float representing the percent of the population that has been infected at this point in time, for each epidemic
r: a float representing the growth rate of the disease,
p: a float representing the extent that r is super or sub exponential
"""

#TODO: Duplicates some code in sir_sim. Move to a higher-level module
disease_trajectory = collections.namedtuple('disease_trajectory', 
                                            ['unique_id', 
                                             'simulation_number', 
                                             'epidemic_number', 
                                             'num_new_infections_over_time', 
                                             'estimated_infections', 
                                             'ground_truth_infections_over_time', 
                                             'total_infections', 
                                             'v', 
                                             'alpha', 
                                             'metadata',
                                             't',
                                             'ground_truth_t',
                                             'cumulative_infections_over_time',
                                             'ground_truth_cumulative_infections_over_time'])  
"""A named tuple to hold a single disease trajectory and all relevant metadata.
    
unique_id: an int representing the unique identifier of each trajectory
simulation_number: an int representing the simulation we generated this trajectory in, only useful for debugging
epidemic_number: an int representing the epidemic we generated this trajectory in, only useful for debugging
num_new_infections_over_time: a np.array of length T representing the number of new infections at each time step up to a threshold
total_estimated_infections: an int representing our estimate of the total infections
ground_truth_infections_over_time: a np.array of length TT representing the *ground truth* number of new infections at each time step
total_infections: an int representing the ground truth number of total infections
total_infections: an int representing the total *ground truth* number of infections
v: a np.array of shape (num_epidemics, num_covariates) and type float representing the covariates used for each epidemic
alpha: a np.array of shape (num_covariates,) and type float representing the weights of each covariate
meta_data: a named tuple of type meta_data
t: a np.array of shape (T,) representing the simulation time of each point in the epidemic
ground_truth_t: a np.array of shape (TT,) representing the simulation time of each point in the ground truth simulation
cumulative_infections_over_time: a np.array representing the "non-predictive" cumsum of the epidemic
ground_truth_cumulative_infections_over_time: a np.array representing the "non-predictive" cumsum of the ground truth simulation
"""  

def generate_ground_truth_VC_curve(pred_final_epidemic_size, r, p):
  """
  Generate the epidemic curve observed to date using the VC model. Assume we
  start with 1 infected individual, and a disease progression based on a
  ViboudChowell model parameterized by r and p.

  Args:
    pred_final_epidemic_size: an int representing the predicted total number 
                              of people to get the disease
    r: a float representing the growth rate of the disease
    p: a float representing the the extent that r is super or sub exponential
  
  Returns:
    list_of_new_infections: a np.array of shape (T,) representing the ground
                            truth number of infected individuals as a function of time
  """
  num_new_infections = 1 # always start with one infection at time 0

  list_of_new_infections = np.array([num_new_infections])

  #While there are still infected people
  while num_new_infections > 0: 
    #Determine the number of new infections
    #By drawing from a poisson distribution 
    total_infected = np.sum(list_of_new_infections)
    frac_infected = float(total_infected)/float(pred_final_epidemic_size)
    mu = r*total_infected**p*(1- frac_infected)
    num_new_infections = stats.poisson.rvs(mu)

    #Record the number of infections that occured at this time point
    list_of_new_infections = np.append(list_of_new_infections, num_new_infections)
    
  return list_of_new_infections

def generate_observed_VC_curves(pred_final_epidemic_size, percent_infected, r, p):
  """
  Generate the epidemic curve observed to date using the VC model.

  Args:
    pred_final_epidemic_size: an int representing the predicted total number 
                              of people to get the disease
    percent_infected: a float representing the percent of the population that 
                  has been infected at this point in time
    r: a float representing the growth rate of the disease
    p: a float representing the the extent that r is super or sub exponential
  
  Returns:
    observed_infections: a np.array of shape (T,) representing the number of
                         newly infected individuals as a function of time, 
                         up to the current time
    ground_truth_infections: an np.array of shape(TT,) representing the number 
                             of newly infectecd individuals over the course of 
                             the *whole* epidemic
  """
  # We require that the total number of infections is >10
  # to eliminate stochastic fadeouts
  total_infections = 0
  # also have a count limit to avoid an infinite loop
  count = 0
    
  while total_infections < 10 and count<5000:
    ground_truth_infections = generate_ground_truth_VC_curve(pred_final_epidemic_size, r, p)
    total_infections = np.sum(ground_truth_infections)
    count += 1

  # calculate the target size for this epidemic
  target_size = pred_final_epidemic_size * percent_infected

  # Find the current time, this is when the cumulative size of the model is still smaller than the target size
  # must be at least 2
  cumulative_infections = np.cumsum(ground_truth_infections)
  current_time = np.max([np.max(np.where(cumulative_infections < target_size)), 2])

  # Save list of infected individuals up until we reach the target_size
  # This is the 'history' of the epidemic up until the current time
  observed_infections = ground_truth_infections[:current_time]

  return observed_infections, ground_truth_infections


def generate_VC_simulations(pred_final_epidemic_size_fn, pred_parameters, num_simulations, num_epidemics, r=1.0, p=0.6):
  """
  Generate many simulations of VC curves.
  Each simulation contains num_epidemics.
  The epidemics may have different covariates, and thus different trajectories.
  However between simulations the covariates are the same, so the only difference 
  is statistical.

  Args:
    pred_final_epidemic_size_fn: a function to generate the predicted final size of the epidemic
    pred_parameters: a tuple containing all the parameters needed by pred_final_epidemic_size_fn
    num_simulations: an int representing the number of simulations to run
    num_epidemics: an int representing the number of epidemics to run in each simulation
    r: a float representing the constant recovery rate (default 1.0)
    p: an int representing the constant population size (default 0.6)
  Returns:
    list_of_disease_trajectory: a list of disease_trajectory named tuples
  """
  list_of_meta_data = [] 
  list_of_disease_trajectory = [] 

  # generate growth rate for all simulations,
  # this is constant between simulations 
  predicted_final_sizes, v, alpha = pred_final_epidemic_size_fn(*pred_parameters)

  # randomly generate the percentage of infected people for each epidemic
  # this is constant between simulations
  #TODO: Make this an argument you can pass in
  percent_infected = np.random.uniform(0.05, 1.0, num_epidemics)
  
  # generate meta data for each epidemic
  # TODO: do better
  for i in range(num_epidemics):
    md = meta_data(num_epidemics, i, predicted_final_sizes[i], percent_infected[i],
                   r, p)
    list_of_meta_data.append(md)

  unique_id = 0
  for j in range(num_simulations):
    for k in range(num_epidemics):
      observed_infections, ground_truth_infections = generate_observed_VC_curves(*list_of_meta_data[k][2:])
      estimated_infections = predicted_final_sizes[i]
      total_infections = np.sum(ground_truth_infections)

      t = np.arange(len(observed_infections))
      ground_truth_t = np.arange(len(ground_truth_infections))
      
      # these are "non-predictive" cumsums.
      # previous day cumsums
      cumulative_infections_over_time = np.cumsum(np.concatenate(([0.], observed_infections)))[:-1]
      ground_truth_cumulative_infections_over_time = np.cumsum(np.concatenate(([0.], ground_truth_infections)))[:-1]

      dt = disease_trajectory(unique_id=unique_id, 
                              simulation_number=j, 
                              epidemic_number=k, 
                              num_new_infections_over_time=observed_infections, 
                              estimated_infections=estimated_infections,
                              ground_truth_infections_over_time=ground_truth_infections,
                              total_infections=total_infections, 
                              v = v[k], 
                              alpha = alpha, 
                              metadata=list_of_meta_data[k],
                              t=t,
                              ground_truth_t = ground_truth_t,
                              cumulative_infections_over_time=cumulative_infections_over_time,
                              ground_truth_cumulative_infections_over_time=ground_truth_cumulative_infections_over_time)
      
      list_of_disease_trajectory.append(dt)
      unique_id += 1
  
  return list_of_disease_trajectory


def pred_final_size_poisson_dist(num_epidemics):
  """
  Predicted final size of the epidemic depend on a single covariate that is randomly generated for each epidemic

  Args:
    num_epidemics: an int representing the number of epidemics to simulate

  Returns:
    final_sizes: a np.array of shape (num_epidemics,) consisting of the final size for each epidemic
    v: a np.array of shape (num_epidemics,) consisting of the randomly generated covariate for each epidemic
    alpha: a np.array of shape (2,) consisting of the weights for each covariate
  """
  v = np.random.uniform(0.0, 3.0, (num_epidemics,))
  alpha = np.array([3, np.log10(3)])
  final_sizes = 10**(alpha[0] + alpha[1]*v)
  return final_sizes, v, alpha