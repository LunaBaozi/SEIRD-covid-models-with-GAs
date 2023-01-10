# -*- coding: utf-8 -*-

from pylab import *

from inspyred import benchmarks
from inspyred.ec import variators
from inspyred_utils import NumpyRandomWrapper
from disk_clutch_brake import DiskClutchBrake, disk_clutch_brake_mutation

import multi_objective

import sys

from functools import reduce

""" 
-------------------------------------------------------------------------
Edit this part to do the exercises

"""

display = True

# parameters for NSGA-2
args = {}
args["pop_size"] = 30 #10
args["max_generations"] = 20 #10
constrained = False

"""
-------------------------------------------------------------------------
"""

problem = DiskClutchBrake(constrained)
if constrained :
    args["constraint_function"] = problem.constraint_function
args["objective_1"] = "Susceptible"  #"Brake Mass (kg)"
args["objective_2"] = "Exposed"      #"Stopping Time (s)"
args["objective_3"] = "Infected" 
args["objective_4"] = "Recovered"

args["variator"] = [variators.blend_crossover,disk_clutch_brake_mutation]

args["fig_title"] = 'NSGA-2'



if __name__ == "__main__" :
    print(len(sys.argv[1]))
    print(args["pop_size"])
    print(args["max_generations"])
    # if len(sys.argv) > 1 :
    #     rng = NumpyRandomWrapper(int(sys.argv[1]))
    # else :
    #     rng = NumpyRandomWrapper()
    rng = NumpyRandomWrapper(int(sys.argv[1]))
    
    final_pop, final_pop_fitnesses = multi_objective.run_nsga2(rng, problem, display=display, 
                                         num_vars=7, **args)
    
    print("Final Population\n", final_pop)
    print()
    print("Final Population Fitnesses\n", final_pop_fitnesses)
    
    output = open("exercise_3.csv", "w")
    for individual, fitness in zip(final_pop, final_pop_fitnesses) :
        output.write(reduce(lambda x,y : str(x) + "," + str(y), 
                            individual))
        output.write(",")
        output.write(reduce(lambda x,y : str(x) + "," + str(y), 
                            fitness))
        output.write("\n")
    output.close()
    
    ioff()
    show()
