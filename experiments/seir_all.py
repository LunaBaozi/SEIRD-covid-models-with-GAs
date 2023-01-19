import os
import math
import pylab
import itertools
from matplotlib import pyplot as plt
from random import Random
from time import time
import inspyred
from scipy.integrate import odeint
import numpy as np
from inspyred.ec.emo import NSGA2
from inspyred.ec import terminators, variators, replacers, selectors

def SEIR_model(z, t, beta, sigma, gamma):
    """
    Reference https://www.idmod.org/docs/hiv/model-seir.html
    """
    S, E, I, R = z
    N = S + E + I + R
    dSdt = -beta*S*I/N
    dEdt = beta*S*I/N - sigma*E
    dIdt = sigma*E - gamma*I
    dRdt = gamma*I
    return [dSdt, dEdt, dIdt, dRdt]

def SEIR_solver(t, initial_conditions, params, infected, recovered):
    initS, initE, initI, initR, initN = initial_conditions
    beta, sigma, gamma = params
    # initS = initN - (initE + initI + initR)

    res = odeint(SEIR_model, [initS, initE, initI, initR], t, args=(beta, sigma, gamma))
    S, E, I, R = res.T
    #print(I)
    rmse_I = np.sqrt(np.mean((I - infected) ** 2))
    rmse_R = np.sqrt(np.mean((R - recovered) ** 2))
    return rmse_I, rmse_R

def generate_parameters(random, args):
    chromosome = []
    bounder = args["_ec"].bounder
    for lo, hi in zip(bounder.lower_bound, bounder.upper_bound):
        chromosome.append(random.uniform(lo, hi))
    return chromosome

def seir_evaluator(self, candidates, args):
    fitness = []
    for c in candidates:
        beta, sigma, gamma, initS, initE = c
        initial_conditions = args["init"]
        initI, initR, initN = initial_conditions
        time = args["time"]
        I = args["I"]
        R = args["R"]
        pop = (initI, initR, initN)

        rmse_I, rmse_R = SEIR_solver(time, (initS, initE, initI, initR, initN), (beta, sigma, gamma), infected=I, recovered=R)

        fitness.append(rmse_I + rmse_R)
    
    return fitness


rand = Random()
rand.seed(int(time()))
# The constraints are as follows:
constraints=((0.1,      0.1,       0.01,    5,   5), 
             (0.2,      0.5,       0.1,     5000,   5000))

algorithm = NSGA2(rand)
algorithm.terminator = terminators.generation_termination
algorithm.observer = inspyred_utils
algorithm.selector = inspyred.ec.selectors.tournament_selection
algorithm.replacer = inspyred.ec.replacers.generational_replacement
algorithm.variator = [inspyred.ec.variators.blend_crossover, inspyred.ec.variators.gaussian_mutation]
projdir = os.path.dirname(os.getcwd())