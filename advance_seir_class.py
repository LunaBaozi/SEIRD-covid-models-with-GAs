from inspyred import benchmarks 
from inspyred.ec.emo import Pareto
from inspyred.ec.variators import mutator
from scipy.integrate import odeint
from pylab import *
import copy

values = [
    np.arange(0.5, 7, 0.001), #r1
    np.arange(0.5, 7, 0.001), #r2
    np.arange(0.01, 1, 0.001), # beta1
    np.arange(0.01, 1, 0.001), # beta2
    np.arange(0.01, 1, 0.001), # gamma
    np.arange(10, 3000, 10), # S
    np.arange(10, 3000, 10) # E
    ]

def ADV_B_SEIR_model(z, t, r1, r2, b1, b2, g, g1, qe, qi, qs):
    alpha = 1/7
    S, Sg, E, Eg, I, Ig, R = z
    N = S + Sg + E + Eg + I + Ig + R
    dSdt = -1*S * (r1*I * (b1+qs-qs*b1) + r2*E* (b2+qs-qs*b2)) / (N - Sg - Eg - Ig)
    dSgdt = S*qs*(r1*I * (1 - b1) + r2*E * (1 - b2)) / (N - Sg - Eg - Ig)
    dEdt = S * (r1*b1*I + r2*b2*E) / (N - Sg - Eg - Ig) - alpha*E*(1 - qe) - E*qe
    dEgdt = E*qe - alpha*Eg
    dIdt = alpha*E * (1 - qe) - I*qi - I*g * (1 - qi)
    dIgdt = Eg*alpha + I*qi - Ig*g1
    dRdt = I*g * (1 - qi) + Ig*g1

    return [dSdt, dSgdt, dEdt, dEgdt, dIdt, dIgdt, dRdt]

def ADV_B_SEIR_solver(t, initial_conditions, params, infected, recovered):
    s0, sg0, e0, eg0, i0, ig0, r0 = initial_conditions
    r1, r2, b1, b2, g, g1, qe, qi, qs = params

    res = odeint(ADV_B_SEIR_model, [s0, sg0, e0, eg0, i0, ig0, r0], t, args=(r1, r2, b1, b2, g, g1, qe, qi, qs))
    S, Sg, E, Eg, I, Ig, R = res.T
    #print(I)
    rmse_I = np.sqrt(np.mean((I - infected) ** 2))
    rmse_R = np.sqrt(np.mean((R - recovered) ** 2))
    return rmse_I, rmse_R

def SEIR_A_solver(t, initial_conditions, params, infected, recovered):
    s0, e0, i0, r0, N = initial_conditions
    r1, r2, beta1, beta2, gamma = params
    res = odeint(SEIR_A_model, [s0, e0, i0, r0, N], t, args=(r1, r2, beta1, beta2, gamma))
    _, _, I, R, _ = res.T
    rmse_I = np.sqrt(np.mean((I - infected) ** 2))
    rmse_R = np.sqrt(np.mean((R - recovered) ** 2))
    return rmse_I, rmse_R

def SEIR_A_model(z, t, r1, r2, beta1, beta2, gamma):
    alpha = 1/7
    S, E, I, R, N = z

    dSdt = -1 * r1 * beta1 * I * S/N - r2 * beta2 * E * S/N
    dEdt = r1 * beta1 * I * S/N + r2 * beta2 * E * S/N - alpha * E
    dIdt = alpha * E - gamma * I
    dRdt = gamma * I
    
    return dSdt, dEdt, dIdt, dRdt, N

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
        benchmarks.Benchmark.__init__(self, len(values), 2)
        self.bounder = SEIRBounder()
        self.maximize = False
        self.constrained=constrained
    
    def generator(self, random, args):
        candidate = []
        for v in values:
            candidate.append(random.uniform(v[0], v[-1]))
        return candidate
    
    def evaluator(self, candidates, args):
        fitness = []
        for c in candidates:
            [r1, r2, beta1, beta2, gamma, initS, initE] = c
            initial_conditions = args["init"]
            initI, initR, initN = initial_conditions
            time = args["time"]
            I = args["I"]
            R = args["R"]
            pop = (initI, initR, initN)

            rmse_I, rmse_R = SEIR_A_solver(time, (initS, initE, initI, initR, initN), (r1, r2, beta1, beta2, gamma), infected=I, recovered=R)
            # TODO how to use the ConstrainedPareto here ??

            if self.constraint_function(c, args) > 0:
                fitness.append([rmse_I, rmse_R])
            else:
                fitness.append([-1, -1])

            # fitness.append([rmse_I + rmse_R])
            # fitness.append(ConstrainedPareto([rmse_I, rmse_R], self.constraint_function(c, args), self.maximize))     
        
        return fitness

    # TODO constraints !!
    def constraint_function(self, candidate, args):
        if not self.constrained :
            return 0
        violations = 0
        [r1, r2, beta1, beta2, gamma, s0, e0] = candidate
        i0, r0, N0 = args['init']
        # s0 = N0 - (e0 + i0 + r0)
        
        if s0+e0+i0+r0 != N0:
            # violations += s0+e0+i0+r0
            return 1
        else:
            return 0
    
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
