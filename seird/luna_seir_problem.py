from inspyred import benchmarks 
from inspyred.ec.emo import Pareto
from inspyred.ec.variators import mutator
from scipy.integrate import odeint
from pylab import *
import copy, math

import read_data


data = read_data.get_data_interval('20200301', '20200401', 22)
infective = data['totale_positivi'].values 
recovered = data['dimessi_guariti'].values
deceased = data['deceduti'].values 

t = np.linspace(0, len(data), len(data))
N = 542166
# e0 = 1/N
# i0 = 0.00
# r0 = 0.00
# s0 = 1 - e0 - i0 - r0
e0 = 1
i0 = 0.00
r0 = 0.00
d0 = 0.00
s0 = N - e0 - i0 - r0 -d0
x0 = [s0, e0, i0, r0, d0]
t_inf = 5.1
# R0 = beta * t_inf


# Here we should also add R0 and others maybe
# values = [
#     np.arange(-0.1, 0.5, 0.001), # beta
#     np.arange(0.1, 0.5, 0.005), # sigma
#     np.arange(0.01, 0.1, 0.001), # gamma
#     ]
values = [
    np.arange(0, 5, 0.01), # beta
    np.arange(0, 1, 0.001), # gamma
    np.arange(0.1, 17, 0.01), # sigma
    np.arange(0.0, 1, 0.001)  # death rate f
    ]


def SEIRD_model(x, t, beta, gamma, sigma, f):   

    beta = beta
    gamma = gamma
    sigma = sigma
    f = f
    S, E, I, R, D = x
    dx = np.zeros(5)
    dx[0] = -beta * S * I /N
    dx[1] = beta * S * I / N - sigma * E
    dx[2] = sigma * E - (1/t_inf) * I
    dx[3] = ((1-f)/t_inf) * I
    dx[4] = (f/t_inf) * I
    
    return dx


def SEIRD_solver(x0, t, beta, gamma, sigma, f):
    
    covid = odeint(SEIRD_model, x0, t, args = (beta, gamma, sigma, f))
    
    return covid 


class SEIRDBounder(object):    
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

class SEIRD(benchmarks.Benchmark):
    
    def __init__(self, constrained=False) : 
        benchmarks.Benchmark.__init__(self, 4, 2)
        self.bounder = SEIRDBounder()
        self.maximize = False
        self.constrained=constrained
    
    def generator(self, random, args):
        return [random.sample(values[i],1)[0] for i in range(self.dimensions)] #[beta, sigma, gamma,S, E]
    
    def evaluator(self, candidates, args):
        fitness = []
        for c in candidates:
            #print(len(c))

            beta, gamma, sigma, f = c
            ret = SEIRD_solver(x0, t, beta, gamma, sigma, f)
            S, E, I, R, D = ret.T
            # print(I - infective)

            # rmse_S = np.sqrt(np.mean((S - susceptible) ** 2))
            # rmse_E = np.sqrt(np.mean((E - exposed) ** 2))
            # rmse_I = np.sqrt(np.mean((I - infective) ** 2))
            rmse_R = np.sqrt(np.mean((R - recovered) ** 2))
            rmse_D = np.sqrt(np.mean((D - deceased) ** 2))

            # f1 = rmse_I
            # f2 = rmse_R
            f1 = rmse_D
            f2 = rmse_R

            fitness.append(ConstrainedPareto([f1, f2],
                                            self.constraint_function(c, S, E, I, R, D),
                                            self.maximize))

        return fitness

   
    def constraint_function(self, candidate, S, E, I, R, D):
        if not self.constrained :
            return 0

        beta, gamma, sigma, f = candidate
        # ret = SEIR_solver(x0, t, beta, gamma, sigma)
        # S, E, I, R = ret.T
        S, E, I, R, D = S, E, I, R, D
        # print(S)

        N_model = S[-1] + E[-1] + I[-1] + R[-1] + D[-1]
        R0 = beta * t_inf
        # print(N_model)

        # dsdt = -beta * S * I / N
        # dedt = beta * S * I / N - sigma * E
        # didt = sigma * E - gamma * I
        # drdt = gamma * I
        

        violations = 0

        if (math.ceil(N_model) != N):
            violations -= (N_model)

        if (R0 >= 7):
            violations -= R0


        return violations

@mutator
def SEIRD_mutation(random, candidate, args):
    mut_rate = args.setdefault('mutation_rate', 0.1)
    bounder = args['_ec'].bounder
    mutant = copy.copy(candidate)
    for i, m in enumerate(mutant):
        if random.random() < mut_rate:
            mutant[i] += random.gauss(0, (values[i][-1] - values[i][0]) / 10.0 )
    mutant = bounder(mutant, args)
    return mutant




