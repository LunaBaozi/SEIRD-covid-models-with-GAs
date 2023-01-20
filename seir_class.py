from inspyred import benchmarks 
from inspyred.ec.emo import Pareto
from inspyred.ec.variators import mutator
from scipy.integrate import odeint
from pylab import *
import copy
from SEIR_models.seir import SEIR_solver

bounds = [
    (0.1, 0.5), # beta (infection rate beta = R0*gamma, R0 about 5.7 or more)
    (0.2, 0.4), # sigma (incubation period, 1/3 to 1/5 about)
    (0.04, 0.3), # gamma (duration of illness 1/18 to 1/5)
    (0, 5000), # E
    (0, 5000), # I
    (0, 5000) # R
    ]

class SEIRBounder(object):
    def __init__(self, bounds):
        self.bounds = bounds
    def __call__(self, candidate, args):
        for i, c in enumerate(candidate):
            candidate[i] = max(self.bounds[i][0], min(self.bounds[i][1], c))
        return candidate

class SEIR(benchmarks.Benchmark):
    
    def __init__(self, constrained=False) : 
        benchmarks.Benchmark.__init__(self, len(bounds), 2)
        self.bounder = SEIRBounder(bounds)
        self.maximize = False
        self.constrained=constrained
    
    def generator(self, random, args):
        candidate = []
        for v in bounds:
            candidate.append(random.uniform(v[0], v[-1]))
        return candidate
    
    def evaluator(self, candidates, args):
        fitness = []
        for c in candidates:
            beta, sigma, gamma, initE, initI, initR = c
            initial_conditions = args["init"]
            _, _, initN = initial_conditions
            time = args["time"]
            I = args["I"]
            R = args["R"]
            initS = initN - initE - initI - initR
            rmse_I, rmse_R = SEIR_solver(time, (initS, initE, initI, initR, initN), (beta, sigma, gamma), infected=I, recovered=R)
 
            fitness.append(Pareto([rmse_I, rmse_R]))     
        
        return fitness

@mutator
def SEIR_mutation(random, candidate, args):
    mut_rate = args.setdefault('mutation_rate', 0.1)
    bounder = args['_ec'].bounder
    mutant = copy.copy(candidate)
    for i, m in enumerate(mutant):
        if random.random() < mut_rate:
            mutant[i] += random.gauss(0, (bounds[i][-1] - bounds[i][0]) / 10.0 )
    mutant = bounder(mutant, args)
    return mutant
