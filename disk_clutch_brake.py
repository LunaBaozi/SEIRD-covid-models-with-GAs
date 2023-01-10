# TODO:
# 1. Define a class for a population made of individuals
# 2. Implement a concrete class to describe our domain problem
# 3. Implement a genetic algorithm
# 4. Run the algorithm
# 5. Tune the algorithm parameters
# NOTE: to use the genetic algorithm to search for SEIR parameters,
# we need to model SEIR parameters as individuals of the population.
# The following 7 parameters are needed:
# 1. alpha
# 2. beta
# 3. epsilon
# 4. gamma
# 5. initial exposed count 
# 6. initial infectious count 
# 7. initial recovered count
# We also need to define value ranges for our 7 parameters
# We also define our fitness function as the RMSE between the true 
# observed fatalities and the predicted fatalities by the model


import copy 

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

from pylab import *

from inspyred import benchmarks
from inspyred.ec import Bounder
from inspyred.ec.emo import Pareto
from inspyred.ec.variators import mutator
from inspyred_utils import NumpyRandomWrapper

import sys
import inspyred_utils
import multi_objective


# DEFINE PARAMETERS

# DEFINE PARAMETERS RANGE
values = [0.0,  # alpha
          0.0,  # beta
          0.0,  # gamma
          0.0,  # epsilon
          0.0,  # initial exposed count
          0.0,  # initial infectious count
          0.0   # initial recovered count
          ]


class DiskClutchBounder(object):
    def __call__(self, candidate, args):
        closest = lambda target, index: min(values[index],
                                            key=lambda x:abs(x-target))
        for i, c in enumerate(candidate):
            candidate[i] = closest(c, i)
        return candidate 

class ConstrainedPareto(Pareto):
    def __init__(self, values=None, violations=None, ec_maximize=True):
        Pareto.__init__(self, values)
        self.violations = violations
        self.ec_maximize = ec_maximize

    def __lt__(self, other):
        if self.violations is None:
            return Pareto.__lt__(self, other)
        elif len(self.values) != len(other.values):
            raise NotImplementedError
        else:
            if self.violations > other.violations:
                return self.ec_maximize
            elif other.violations > self.violations:
                return not self.ec_maximize
            elif self.violations > 0:
                return False
            else:
                not_worse = True
                strictly_better = False
                for x, y, m in zip(self.values, other.values, self.maximize):
                    if m:
                        if x > y:
                            not_worse = False
                        elif y > x:
                            strictly_better = True
                        else:
                            if x < y:
                                not_worse = False
                            elif y < x:
                                strictly_better = True
            return not_worse and strictly_better


class DiskClutchBrake(benchmarks.Benchmark):
    def __init__(self, constrained=False):
        benchmarks.Benchmark.__init__(self, 5, 2)
        self.bounder = DiskClutchBounder()
        self.maximize = False
        self.constrained = constrained

    def generator(self, random, args):
        return [random.sample(values[i], 1)[0] for i in range(self.dimensions)]

    def evaluator(self, candidates, args):
        fitness = []
        for c in candidates:
            f1 = -(1-u) * beta * s * i
            f2 = (1-u) * beta * s * i - (alpha * e)
            f3 = (alpha * e) - (gamma * i)
            f4 = gamma * i

            fitness.append(ConstrainedPareto([f1, f2, f3, f4],
                                              self.constraint_function(c),
                                              self.maximize))

        return fitness 

    def constraint_function(self, candidate):
        if not self.constrained:
            return 0
        """Return the magnitude of constraint violations."""
        violations = 0

        return violations 

def disk_clutch_brake_mutation(random, candidate, args):
    mut_rate = args.setdefault('mutation rate', 0.1)
    bounder = args['_ec'].bounder
    mutant = copy.copy(candidate)
    for i, m in enumerate(mutant):
        if random.random() < mut_rate:
            mutant[i] += random.gauss(0, (values[i][-1] - values[i][0]) / 10.0)
        mutant = bounder(mutant, args)
        return mutant










# class MyProblem(ElementwiseProblem):
    
#     def __init__(self):
#         super().__init__(n_var=7,
#                          n_obj=4,
#                          n_ieq_constr=4,
#                          xl=np.array([-10, 10]),
#                          xu=np.array([10,10]))
        # benchmarks.Benchmark.__init__(self, dimensions, len(objectives))
        # self.bounder = Bounder([-5.0] * self.dimensions, [5.0] * self.dimensions)
        # self.maximize = False
        # self.evaluators = [cls(dimensions).evaluator for cls in objectives]
    
    # def generator(self, random, args):
    #     return [random.uniform(-5.0, 5.0) for _ in range(self.dimensions)]
        
    # def evaluator(self, candidates, args):
    #     fitness = [evaluator(candidates, args) for evaluator in self.evaluators]
    #     return list(map(Pareto, zip(*fitness)))

    # x = np.array([u, N, beta, s, e, i, r, alpha, gamma, R0])
    
    # def _evaluate(self, x, out, *args, **kwargs):

    #     f1 = -(1-u) * beta * s * i
    #     f2 = (1-u) * beta * s * i - (alpha * e)
    #     f3 = (alpha * e) - (gamma * i)
    #     f4 = gamma * i

    #     g1 = 1/N
    #     g2 = 0.00
    #     g3 = 0.00
    #     g4 = 1 - e0 - i0 - r0 

    #     out["F"] = [f1, f2, f3, f4]
    #     out["G"] = [g1, g2, g3, g4]
    
    
# objective function RMSD wrt real data distribution

# class MyProblem(ElementwiseProblem):

#     def __init__(self):
#         super().__init__(n_var=2,
#                          n_obj=2,
#                          n_ieq_constr=2,
#                          xl=np.array([-2,-2]),
#                          xu=np.array([2,2]))

#     def _evaluate(self, x, out, *args, **kwargs):
#         f1 = 100 * (x[0]**2 + x[1]**2)
#         f2 = (x[0]-1)**2 + x[1]**2

#         g1 = 2*(x[0]-0.1) * (x[0]-0.9) / 0.18
#         g2 = - 20*(x[0]-0.4) * (x[0]-0.6) / 4.8

#         out["F"] = [f1, f2]
#         out["G"] = [g1, g2]


# problem = MyProblem()
""" 
-------------------------------------------------------------------------
Edit this part to do the exercises

"""

# display = True # Plot initial and final populations
# #num_vars = 2 # set 3 for Kursawe, set to 19+num_objs for DTLZ7
# num_objs = 3 # used only for DTLZ7
# num_vars = 19 + num_objs

# # parameters for NSGA-2
# args = {}
# args["pop_size"] = 60*10e7
# args["max_generations"] = 20

# problem = benchmarks.Kursawe(num_vars) # set num_vars = 3
#problem = benchmarks.DTLZ7(num_vars,num_objs) # set num_objs = 3 and num_vars = 19+num_objs

#problem = MyBenchmark(num_vars, [benchmarks.Rastrigin, benchmarks.Schwefel] )
#problem = MyBenchmark(num_vars, [benchmarks.Sphere, benchmarks.Rastrigin, benchmarks.Schwefel] )

"""
-------------------------------------------------------------------------
"""

# x = np.array([u, N, beta, s, e, i, r, alpha, gamma, R0])

# u = 0.3  # social distancing (0-1)
#          # 0 = no social distancing
#          # 0.1 = masks
#          # 0.2 = masks and hybrid classes
#          # 0.3 = masks, hybrid and online classes

# t_incubation = 5.1
# t_infective = 3.3
# R0 = 2.4
# N = 33517  # number of students

# # initial number of infected and recovered individuals
# e0 = 1/N
# i0 = 0.00
# r0 = 0.00
# s0 = 1 - e0 - i0 - r0 
# x0 = [s0, e0, i0, r0]

# alpha = 1/t_incubation
# gamma = 1/t_infective
# beta = R0*gamma

# def covid(x, t):
#     s, e, i, r = x 
#     dx = np.zeros(4)
#     dx[0] = -(1-u) * beta * s * i
#     dx[1] = (1-u) * beta * s * i - (alpha * e)
#     dx[2] = (alpha * e) - (gamma * i)
#     dx[3] = gamma * i
#     return dx

# t = np.linspace(0, 200, 101)
# x = odeint(covid, x0, t)
# s = x[:, 0]; e = x[:, 1]; i = x[:, 2]; r = x[:, 3]

# # plot the data
# plt.figure(figsize=(8, 5))

# plt.subplot(2, 1, 1)
# plt.title('Social distancing = ' + str(u*100) + '%')
# plt.plot(t, s, color='blue', lw=3, label='Susceptible')
# plt.plot(t, r, color='red', lw=3, label='Recovered')
# plt.ylabel('Fraction')
# plt.legend()

# plt.subplot(2, 1, 2)
# plt.plot(t, i, color='orange', lw=3, label='Infective')
# plt.plot(t, e, color='purple', lw=3, label='Exposed')
# plt.ylim(0, 0.2)
# plt.xlabel('Time (days)')
# plt.ylabel('Fraction')
# plt.legend()

# plt.show()






# args["fig_title"] = 'NSGA-2'
    
# if __name__ == "__main__" :
#     if len(sys.argv) > 1 :
#         rng = NumpyRandomWrapper(int(sys.argv[1]))
#     else :
#         rng = NumpyRandomWrapper()
    
#     final_pop, final_pop_fitnesses = multi_objective.run_nsga2(rng, problem,
#                                         display=display, num_vars=num_vars,
#                                         **args)
    
#     print("Final Population\n", final_pop)
#     print()
#     print("Final Population Fitnesses\n", final_pop_fitnesses)
    
#     ioff()
#     show()