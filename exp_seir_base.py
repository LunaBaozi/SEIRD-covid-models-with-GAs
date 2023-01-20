from pylab import *
import read_data
import pandas as pd
import numpy as np
from scipy.integrate import odeint
import inspyred
import random
from inspyred_utils import NumpyRandomWrapper
from seir_class import SEIR, SEIR_mutation
import sys
from inspyred.ec import variators
import seir_objective
from seir_model_plot_utils import plot_difference
import pso

# - DEFINITION OF THE INITIAL GA PARAMETERS
args = {}
args["pop_size"] = 20
args["max_generations"] = 200
# ------------------------------------------

display = True
constrained = False

if __name__ == '__main__':
    # - DATA ACQUISITION (yyyymmdd)
    cases = read_data.get_data_interval('20200301', '20200401', 22)
    print(cases)
    cases['data'] = pd.to_datetime(cases['data'])
    data = cases.sort_values('data')
    data['days'] = (data['data'] - data['data'].min()).dt.days
    args['time'] = data['days'].values 
    
    tot_pos = data['totale_positivi'].values
    tot_rec = data['dimessi_guariti'].values
    tot_dec = data['deceduti'].values
    # ---------------------------------------
    
    args["variator"] = [variators.blend_crossover, SEIR_mutation]   
    args["fig_title"] = 'NSGA-2'

    # - INITIAL CONDITIONS
    N = 542166 # OFFICIAL NUMBER OF POPULATION IN P.A. TRENTO IN 2020 (ISTAT)
    I_0 = tot_pos[0] # NUMBER OF INITIAL INFECTED 
    R_0 = tot_rec[0] # NUMBER OF INITIAL RECOVERED 
    D_0 = tot_dec[0] # NUMBER OF INITIAL DEATH 

    # args["fitness_weights"] = [0.5, 1.5]

    args['init'] = (I_0, R_0, N)
    args['I'] = tot_pos
    args['R'] = tot_rec

    rng = random.Random(0)
    problem = SEIR(constrained)

    if len(sys.argv) > 1 :
        rng = NumpyRandomWrapper(int(sys.argv[1]))
    else :
        rng = NumpyRandomWrapper()

    final_pop, final_pop_fitnesses = seir_objective.run_nsga2(rng, problem, display=display, 
                                         num_vars=7, **args)

    print("Final Population\n", final_pop)
    print()
    print("Final Population Fitnesses\n", final_pop_fitnesses)

    ioff()
    show()

    # - DISPLAY THE REAL DATA AND THE JUST COMPUTED MODEL
    # - HERE WE DECIDE TO SHOW HOW THE MODEL WILL DIFFER FROM THE REAL DATA AFTER 30 DAYS MORE
    cases = read_data.get_data_interval('20200301', '20200601', 22)
    cases['data'] = pd.to_datetime(cases['data'])
    data = cases.sort_values('data')
    data['days'] = (data['data'] - data['data'].min()).dt.days
    plot_difference(data, final_pop[0], N)