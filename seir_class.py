from inspyred import benchmarks 
from inspyred.ec.emo import Pareto
from inspyred.ec.variators import mutator
from scipy.integrate import odeint
from pylab import *
import copy

MIN_BETA = 1e-3
MAX_BETA = 1.0
MIN_INCUBATION_PERIOD = 2.0
MAX_INCUBATION_PERIOD = 14
MIN_INFECTIOUS_PERIOD = 3.0
# parameters
# TODO
#possible values
# TODO check the possible range of gamma beta and sigma
values = [arange(0.1,0.2,0.01), arange(0.1,0.5,0.05), arange(0.01,0.1,0.01), arange(0, 3000, 10)]

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
    initE, initI, initR, initN = initial_conditions
    beta, sigma, gamma = params
    initS = initN - (initE + initI + initR)

    res = odeint(SEIR_model, [initS, initE, initI, initR], t, args=(beta, sigma, gamma))
    S, E, I, R = res.T
    #print(I)
    rmse_I = np.sqrt(np.mean((I - infected) ** 2))
    rmse_R = np.sqrt(np.mean((R - recovered) ** 2))
    return rmse_I, rmse_R

class SEIRBounder(object):    
    def __call__(self, candidate, args):
        closest = lambda target, index: min(values[index], 
                                            key=lambda x: abs(x-target))        
        for i, c in enumerate(candidate):
            candidate[i] = closest(c,i)
        return candidate

class ConstrainedPareto(Pareto):
    def __init__(self, values=None, violations=None, ec_maximize=True):
        Pareto.__init__(self, values)
        self.violations = violations
        self.ec_maximize = ec_maximize
    
    def __lt__(self, other):
        if self.violations is None :
            return Pareto.__lt__(self, other)
        elif len(self.values) != len(other.values):
            raise NotImplementedError
        else:
            if self.violations > other.violations :
                # if self has more violations than other
                # return true if EC is maximizing otherwise false 
                return (self.ec_maximize)
            elif other.violations > self.violations :
                # if other has more violations than self
                # return true if EC is minimizing otherwise false  
                return (not self.ec_maximize)
            elif self.violations > 0 :
                # if both equally infeasible (> 0) than cannot compare
                return False
            else :
                # only consider regular dominance if both are feasible
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

class SEIR(benchmarks.Benchmark):
    
    def __init__(self, constrained=False) : 
        benchmarks.Benchmark.__init__(self, 4, 1) # TODO come inzializzarlo?? 
        self.bounder = SEIRBounder()
        self.maximize = False
        self.constrained=constrained
    
    def generator(self, random, args):
        beta = random.uniform(1e-3, 5.0) 
        sigma = random.uniform(0.05, 0.5) 
        gamma = random.uniform(0.05, 0.5)
        E = random.uniform(0, 3000)
        return [beta, sigma, gamma, E]
    
    def evaluator(self, candidates, args):
        fitness = []
        for c in candidates:
            beta, sigma, gamma, initE = c
            initial_conditions = args["init"]
            initI, initR, initN = initial_conditions
            time = args["time"]
            I = args["I"]
            R = args["R"]
            pop = (initI, initR, initN)

            rmse_I, rmse_R = SEIR_solver(time, (initE, initI, initR, initN), (beta, sigma, gamma), infected=I, recovered=R)
            # TODO how to use the ConstrainedPareto here ??

            #fitness.append([rmse_I])
            fitness.append(ConstrainedPareto([rmse_I, rmse_R], self.constraint_function(c, args), self.maximize))     
        
        return fitness

    # TODO constraints !!
    def constraint_function(self, candidate, args):
        if not self.constrained :
            return 0
        violations = 0
        beta, sigma, gamma, e0 = candidate
        i0, r0, N0 = args['init']
        s0 = N0 - (e0 + i0 + r0)
        
        if s0+e0+i0+r0 != N0:
            violations -= abs(s0+e0+i0+r0)
            print(violations)
    
        return violations

@mutator
def SEIR_mutation(random, candidate, args):
    mut_rate = args.setdefault('mutation_rate', 0.1)
    bounder = args['_ec'].bounder
    mutant = copy.copy(candidate)
    for i, m in enumerate(mutant):
        if random.random() < mut_rate:
            mutant[i] += random.gauss(0, (values[i][-1] - values[i][0]) / 10.0 )
    mutant = bounder(mutant, args)
    return mutant
