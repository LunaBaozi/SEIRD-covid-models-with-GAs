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
e0 = 1
i0 = infective[0]
r0 = recovered[0]
d0 = deceased[0]       
s0 = N - e0 - i0 - r0 -d0
x0 = [s0, e0, i0, r0, d0]
t_inf = 5.1


values = [
    np.arange(0, 5, 0.01), # beta
    np.arange(0, 1, 0.001), # gamma
    np.arange(0.1, 17, 0.01), # sigma
    np.arange(0.0, 1, 0.001)  # death rate f
    ]


def SEIRD_model(x, t, beta, gamma, sigma, f):   
    """
    Define the SEIRD model to use.
    :param x:       array of initial conditions
    :param t:       array of time frame
    :param beta:    infection rate
    :param gamma:   inverse average recovery time
    :param sigma:   inverse average infection latent time
    :param f:       death rate
    :return:        array containing estimated values
    """
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
    """
    Helper function to solve the ODE.
    """
    covid = odeint(SEIRD_model, x0, t, args = (beta, gamma, sigma, f))
    
    return covid 


class SEIRDBounder(object):    
    """
    Function to assign values in range to parameters.
    """
    def __call__(self, candidate, args):
        closest = lambda target, index: min(values[index], 
                                            key=lambda x: abs(x-target))        
        for i, c in enumerate(candidate):
            candidate[i] = closest(c,i)
        return candidate


class ConstrainedPareto(Pareto):
    """
    Function to compute the constrained Pareto frontier of solution.
    Function logic directly comes from Course Lab's experiments.
    """
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
                return (self.ec_maximize)
            elif other.violations > self.violations :
                return (not self.ec_maximize)
            elif self.violations > 0 :
                return False
            else :
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
    """
    Definition of problem class, inherits from Benchmark.
    Defines the class, the generator, the evaluator and
    the constraints of the problem.
    """
    def __init__(self, constrained=False) : 
        benchmarks.Benchmark.__init__(self, 4, 2)
        self.bounder = SEIRDBounder()
        self.maximize = False
        self.constrained=constrained
    
    def generator(self, random, args):
        return [random.sample(values[i],1)[0] for i in range(self.dimensions)] 
    
    def evaluator(self, candidates, args):
        fitness = []
        for i, c in enumerate(candidates):
            beta, gamma, sigma, f = c
            ret = SEIRD_solver(x0, t, beta, gamma, sigma, f)
            S, E, I, R, D = ret.T
            rmse_I = np.sqrt(np.mean((I - infective) ** 2))
            rmse_R = np.sqrt(np.mean((R - recovered) ** 2))
            rmse_D = np.sqrt(np.mean((D - deceased) ** 2))

            f1 = rmse_D
            f2 = rmse_R
            f3 = rmse_I

            fitness.append(ConstrainedPareto([f1, f2, f3],
                                            self.constraint_function(c, S, E, I, R, D),
                                            self.maximize))

        return fitness

   
    def constraint_function(self, candidate, S, E, I, R, D):

        if not self.constrained :
            return 0

        beta, gamma, sigma, f = candidate
        N_model = S[-1] + E[-1] + I[-1] + R[-1] + D[-1]
        R0 = beta * t_inf

        violations = 0

        if (math.ceil(N_model) != N):
            violations -= (N_model)
            print(f'Violation N_model = {N_model} added')

        if (R0 >= 2):
            violations -= R0
            print(f'Violation R0 = {R0} added')

        if (D >= 15 * N /100):
            violations -= D
            print(f'Violation deaths = {D} added')

        if (I <= 10 * N / 100):
            violations -= I 
            print(f'Violation I = {I} added')

        return violations


@mutator
def SEIRD_mutation(random, candidate, args):
    """
    Defines the mutation function.
    :param random:      random value
    :param candidate:   solution candidate
    :return:            mutated solution candidate
    """
    mut_rate = args.setdefault('mutation_rate', 0.1)
    bounder = args['_ec'].bounder
    mutant = copy.copy(candidate)
    for i, m in enumerate(mutant):
        if random.random() < mut_rate:
            mutant[i] += random.gauss(0, (values[i][-1] - values[i][0]) / 10.0 )
    mutant = bounder(mutant, args)
    return mutant




