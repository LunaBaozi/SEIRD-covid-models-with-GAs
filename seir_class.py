from inspyred import benchmarks 
from inspyred.ec.emo import Pareto
from inspyred.ec.variators import mutator
from scipy.integrate import odeint
from pylab import *
import copy

# parameters
# TODO
#possible values
# TODO check the possible range of gamma beta and sigma
values = [arange(0.1,0.2,0.01), arange(0.1,0.5,0.05), arange(0.01,0.1,0.01)]

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
        self.ec_maximize=ec_maximize
    
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
        benchmarks.Benchmark.__init__(self, 4, 2) # TODO come inzializzarlo?? 
        self.bounder = SEIRBounder()
        self.maximize = False
        self.constrained=constrained
    
    def generator(self, random, args):
        beta = random.uniform(0.05, 0.5) 
        sigma = random.uniform(0.01, 0.1) 
        gamma = random.uniform(0.01, 0.2)
        return [beta, sigma, gamma]
    
    def evaluator(self, candidates, args):
        fitness = []
        for c in candidates:
            beta, sigma, gamma = c
            initial_conditions = args["init"]
            time = args["time"]
            I = args["I"]
            R = args["R"]

            rmse_I, rmse_R = SEIR_solver(time, initial_conditions, (beta, sigma, gamma), infected=I, recovered=R)
            # TODO how to use the ConstrainedPareto here ??
            # fitness.append((rmse_I, rmse_R))
            fitness.append(ConstrainedPareto([rmse_I, rmse_R], self.constraint_function(c, args), self.maximize))     
        
        return fitness

    # TODO constraints !!
    def constraint_function(self, candidate, args):
        if not self.constrained :
            return 0
        N = args["N"]
        S, E, I, R = candidate
        # constrain 1: S + E + I + R = N (total population)
        constrain1 = abs(S + E + I + R - N)
        # constrain 2: E, I, R >= 0
        constrain2 = max(E, I, R, 0)
        return constrain1, constrain2

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
