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

args = {}
args["pop_size"] = 30
args["max_generations"] = 100

display = True
constrained = False  # True

if __name__ == '__main__':
    cases = read_data.get_data_interval('20200301', '20200401', 22)
    print(cases)
    cases['data'] = pd.to_datetime(cases['data'])
    data = cases.sort_values('data')
    data['days'] = (data['data'] - data['data'].min()).dt.days
    args['time'] = data['days'].values 
    
    tot_pos = data['totale_positivi'].values
    tot_rec = data['dimessi_guariti'].values
    tot_dec = data['deceduti'].values

    args["variator"] = [variators.blend_crossover, SEIR_mutation]   
    args["fig_title"] = 'NSGA-2'

    #initial conditions
    N = 542166
    I_0 = tot_pos[0] - tot_rec[0] - tot_dec[0]
    R_0 = tot_rec[0]
    D_0 = tot_dec[0]
    E_0 = N - (I_0 + R_0 + D_0)

    args['init'] = (E_0, I_0, R_0, N)
    args['I'] = tot_pos
    args['R'] = tot_rec

    rng = random.Random(0)

    problem = SEIR(constrained)

    if len(sys.argv) > 1 :
        rng = NumpyRandomWrapper(int(sys.argv[1]))
    else :
        rng = NumpyRandomWrapper()

    final_pop, final_pop_fitnesses = seir_objective.run_nsga2(rng, problem, display=display, 
                                         num_vars=3, **args)

    print("Final Population\n", final_pop)
    print()
    print("Final Population Fitnesses\n", final_pop_fitnesses)

    ioff()
    show()