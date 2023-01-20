from pylab import *

from inspyred.ec import variators
from inspyred_utils import NumpyRandomWrapper
from luna_seir_problem import SEIRD, SEIRD_mutation
import luna_multiobjective as luna_multiobjective
import read_data
from luna_nuovoplot import plotting_function_model

import sys

from functools import reduce


data = read_data.get_data_interval('20200301', '20200401', 22)
infective = data['totale_positivi'].values 
recovered = data['dimessi_guariti'].values
deceased = data['deceduti'].values 

N = 542166
num_vars = 3

display = False
args = {}
args["pop_size"] = 30
args["max_generations"] = 100
args["fitness_weights"] = [0.9, 0.1]

display = True
constrained = True


problem = SEIRD(constrained)
if constrained:
    args['constraint_function'] = problem.constraint_function

args["variator"] = [variators.blend_crossover, SEIRD_mutation]   
args["fig_title"] = 'NSGA-2'

if __name__ == '__main__':
    if len(sys.argv) > 1:
        rng = NumpyRandomWrapper(int(sys.argv[1]))
    else:
        rng = NumpyRandomWrapper()

    
    best_individual, best_fitness = luna_multiobjective.run_ga(rng, problem, display=display, 
                                         num_vars=num_vars, **args)

    print("Best Individual", best_individual)
    print("Best Fitness", best_fitness)
    
    plotting_function_model(best_individual)
    if display :    
        ioff()
        show()

   
