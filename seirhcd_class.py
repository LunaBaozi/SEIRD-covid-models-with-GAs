from inspyred import benchmarks 
from inspyred.ec.emo import Pareto
from inspyred.ec.variators import mutator
from scipy.integrate import odeint
from pylab import *
import copy
from seirhdc import SEIR_HCD_model

#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------

def SEIRHCD_solver(t, initial_conditions, params, infected, recovered):

    R_t, t_inc, t_inf, t_hosp0, t_crit0, m_a0, c_a0, f_a0 = params
    res = SEIR_HCD_model(t, initial_conditions, R_t, t_inc, t_inf, t_hosp0, t_crit0, m_a0, c_a0, f_a0, decay_values=False)
    [S_out, E_out, I_out, R_out, H_out, C_out, D_out] = res

    rmse_I = np.sqrt(np.mean((I_out - infected) ** 2))
    rmse_R = np.sqrt(np.mean((R_out - recovered) ** 2))
    return rmse_I, rmse_R

#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------
values = [arange(1,5000,10),
          arange(1,5000,10), 
          arange(1,5000,10), 
          arange(1,5000,10), 
          arange(1, 10, 0.1), 
          arange(1,13,0.1), 
          arange(1, 10, 0.1), 
          arange(1, 10, 0.1), 
          arange(1, 20, 0.1), 
          arange(0.1, 3, 0.01), 
          arange(0.01, 0.5, 0.01), 
          arange(0.1, 0.6, 0.01)]

class SEIRHCD(benchmarks.Benchmark):
    
    def __init__(self, constrained=False) : 
        benchmarks.Benchmark.__init__(self, 12, 1) 
        self.maximize = False
        self.constrained=constrained
    
    def generator(self, random, args):
        S = random.uniform(1, 5000)
        E = random.uniform(1, 5000)
        H = random.uniform(1, 5000)
        C = random.uniform(1, 5000)
        R_t = random.uniform(0.1, 10)
        t_inc = random.uniform(1, 13)
        t_inf = random.uniform(1, 10)
        t_hosp0 = random.uniform(1, 10)
        t_crit0 = random.uniform(1, 20)
        m_a0 = random.uniform(0.1, 3)
        c_a0 = random.uniform(0.01, 0.5)
        f_a0 = random.uniform(0.1, 0.6)
        return [S, E, H, C, R_t, t_inc, t_inf, t_hosp0, t_crit0, m_a0, c_a0, f_a0]
    
    def evaluator(self, candidates, args):
        fitness = []
        for c in candidates:
            initS, initE, initH, initC, R_t, t_inc, t_inf, t_hosp0, t_crit0, m_a0, c_a0, f_a0 = c
            initial_conditions = args["init"]
            initI, initR, initD, initN = initial_conditions
            time = args["time"]
            I = args["I"]
            R = args["R"]

            rmse_I, rmse_R = SEIRHCD_solver(time, 
                (initS, initE, initI, initR, initH, initC, initD), 
                (R_t, t_inc, t_inf, t_hosp0, t_crit0, m_a0, c_a0, f_a0),
                infected=I, recovered=R)

            fitness.append([rmse_I])
        
        return fitness

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
