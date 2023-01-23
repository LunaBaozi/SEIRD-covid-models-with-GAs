from pylab import *
import pandas as pd
import random
from utils.inspyred_utils import NumpyRandomWrapper
from utils.seird_class import SEIRD, SEIR_mutation
import sys
from inspyred.ec import variators
from utils.seir_objective import run_nsga2, run_ga
from utils.seir_model_plot_utils import plot_difference_seird
from utils.read_data import get_data_interval

args = {}
args["pop_size"] = 70
args["max_generations"] = 300

constrained = True

if __name__ == '__main__':
    cases = get_data_interval('20210301', '20210401', 22)
    print(cases)
    cases['data'] = pd.to_datetime(cases['data'])
    data = cases.sort_values('data')
    data['days'] = (data['data'] - data['data'].min()).dt.days
    args['time'] = data['days'].values 
    
    tot_pos = data['totale_positivi'].values
    tot_rec = data['dimessi_guariti'].values
    tot_dec = data['deceduti'].values

    args["variator"] = [variators.blend_crossover, SEIR_mutation]   
    args["fig_title"] = 'Plot - EA'

    #initial conditions
    N = 542166
    I_0 = tot_pos[0]
    R_0 = tot_rec[0]
    D_0 = tot_dec[0]

    args['init'] = (I_0, R_0, D_0, N)
    args['I'] = tot_pos
    args['R'] = tot_rec
    args['D'] = tot_dec

    rng = random.Random(0)

    problem = SEIRD(constrained)

    if len(sys.argv) > 1 :
        rng = NumpyRandomWrapper(int(sys.argv[1]))
    else :
        rng = NumpyRandomWrapper()

    # COMMENT THIS PART TO USE THE STANDARD GA
    final_pop, final_pop_fitnesses = run_nsga2(rng, problem, display=False, 
                                        num_vars=5, **args)

    print("Final Population\n", final_pop)
    print()
    print("Final Population Fitnesses\n", final_pop_fitnesses)

    ioff()
    show()

    cases = get_data_interval('20210301', '20210601', 22)
    cases['data'] = pd.to_datetime(cases['data'])
    data = cases.sort_values('data')
    data['days'] = (data['data'] - data['data'].min()).dt.days
    plot_difference_seird(data, final_pop[0], N)
    #--------------------------------------------------------------------

    # UNCOMMENT IF YOU WANT TO USE THE STANDARD GA
    # final_pop, final_pop_fitnesses = seir_objective.run_ga(rng, problem, display=True, 
    #                                      num_vars=5, **args)

    # print("Final Population\n", final_pop)
    # print()
    # print("Final Population Fitnesses\n", final_pop_fitnesses)

    # ioff()
    # show()

    # cases = get_data_interval('20210301', '20210601', 22)
    # cases['data'] = pd.to_datetime(cases['data'])
    # data = cases.sort_values('data')
    # data['days'] = (data['data'] - data['data'].min()).dt.days
    # plot_difference_seird(data, final_pop, N)
    #--------------------------------------------------------------------