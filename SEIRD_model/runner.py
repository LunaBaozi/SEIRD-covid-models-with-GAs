from pylab import *

from inspyred.ec import variators
from inspyred_utils import NumpyRandomWrapper
from seird_problem import SEIRD, SEIRD_mutation
import multiobjective as multiobjective
import read_data
from plot_utils import plotting_function_model

import sys
import pandas as pd

from functools import reduce


name = 'test_D_model'
data = read_data.get_data_interval('20200301', '20200401', 22)
infective = data['totale_positivi'].values 
recovered = data['dimessi_guariti'].values
deceased = data['deceduti'].values 

N = 542166
num_vars = 4

display = False
args = {}
args["pop_size"] = 30
args["max_generations"] = 100
# f1 is death, f2 is recovered, f3 is infectious
args["fitness_weights"] = [0.8, 0.1, 0.1]

display = True
constrained = False


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

    # # Uncomment the following code to try Single-Objective Optimization
    # best_individual, best_fitness = multiobjective.run_ga(rng, problem, display=display, 
    #                                      num_vars=num_vars, **args)

    # print("Best Individual", best_individual)
    # print("Best Fitness", best_fitness)
    
    # plotting_function_model(best_individual)
    # if display :    
    #     ioff()
    #     show()


    # # Uncomment the following code to try Multi-Objective Optimization
    final_pop, final_pop_fitnesses = multiobjective.run_nsga2(rng, problem, display=display, 
                                         num_vars=num_vars, **args)
    
    print("Final Population\n", final_pop)
    print()
    print("Final Population Fitnesses\n", final_pop_fitnesses)
    
    output = open(f"{name}.csv", "w")
    for individual, fitness in zip(final_pop, final_pop_fitnesses) :
        output.write(reduce(lambda x,y : str(x) + "," + str(y), 
                            individual))
        output.write(",")
        output.write(reduce(lambda x,y : str(x) + "," + str(y), 
                            fitness))
        output.write("\n")
    output.close()
    

    dataframe=pd.DataFrame(final_pop, columns=['beta', 'gamma', 'sigma', 'f']) 
    plotting_function_model(dataframe)

    ioff()
    show()

   
