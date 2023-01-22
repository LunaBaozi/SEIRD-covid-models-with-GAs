from inspyred import benchmarks 
from inspyred.ec.emo import Pareto
from inspyred.ec.variators import mutator
from SEIR_models.seird import SEIRD_solver
from pylab import *
import copy
import math

bounds = [
    (1e-4, 1), # alpha (infection fatality rate)
    (1e-4, 1), # beta (infection rate beta = R0*gamma, R0 about 5.7 or more)
    (0.1, 17), # sigma (incubation period, 1/3 to 1/5 about)
    (1e-5, 1), # gamma (duration of illness 1/18 to 1/5)
    (0, 50000) # E
    ]

# bounds = [
#     (8e-5, 1e-4), # alpha (infection fatality rate)
#     (5e-3, 1.), # beta (infection rate beta = R0*gamma, R0 about 5.7 or more)
#     (1/5.6, 1/4.8), # sigma (incubation period, 1/3 to 1/5 about)
#     (1e-5, 1), # gamma (duration of illness 1/18 to 1/5)
#     (0, 50000), # E
#     ]

class SEIRBounder(object):
    def __init__(self, bounds):
        self.bounds = bounds
    def __call__(self, candidate, args):
        for i, c in enumerate(candidate):
            candidate[i] = max(self.bounds[i][0], min(self.bounds[i][1], c))
        return candidate

class SEIRD(benchmarks.Benchmark):
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
            alpha, beta, sigma, gamma, initE = c
            initial_conditions = args["init"]
            initI, initR, initD, initN = initial_conditions
            time = args["time"]
            infected = args["I"]
            recovered = args["R"]
            deceased = args["D"]
            initS = initN - initE - initI - initR - initD
            res = SEIRD_solver(time, (initS, initE, initI, initR, initD, initN), (alpha, beta, sigma, gamma))
            S, E, I, R, D, _ = res.T
            rmse_I = np.sqrt(np.mean((I - infected) ** 2))
            rmse_R = np.sqrt(np.mean((R - recovered) ** 2))
            rmse_D = np.sqrt(np.mean((D - deceased) ** 2))

            # COMMENT THIS PART FOR THE STANDARD GA
            constraint_violation = self.constraint_function(c, S, E, I, R, D, initN)
            fitness.append((rmse_D + rmse_I, rmse_R, constraint_violation))
            # ---------------------------------------------------------------------------

            # UNCOMMENT IF YOU WANT TO USE THE STANDARD GA
            # fitness.append(rmse_D  + rmse_I + rmse_D)

        return fitness

    def constraint_function(self, candidate, S, E, I, R, D, N):
        if not self.constrained :
            return 0

        N_model = S[-1] + E[-1] + I[-1] + R[-1] + D[-1]
        return abs(math.ceil(N_model) - N)

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
