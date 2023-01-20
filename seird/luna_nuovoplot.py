import numpy as np
import matplotlib.pyplot as plt

import read_data
from luna_seir_problem import SEIRD_solver

data = read_data.get_data_interval('20200301', '20200401', 22)
infective = data['totale_positivi'].values 
recovered = data['dimessi_guariti'].values
deceased = data['deceduti'].values 

N = 542166
num_vars = 4

t = np.linspace(0, len(data), len(data))
N = 542166


def plotting_function_real(N, t):

    I = infective 
    R = recovered 
    D = deceased 

    # Plot the data on three separate curves for S(t), I(t) and R(t)
    fig = plt.figure(facecolor='w')
    ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
    ax.plot(t, D/N * 100, 'b', alpha=0.5, lw=2, label='Deceased')
    ax.plot(t, I/N * 100, 'r', alpha=0.5, lw=2, label='Infected')
    ax.plot(t, R/N * 100, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
    ax.set_xlabel('Time /days')
    ax.set_ylabel('Percentage pop number')
    # ax.set_ylim(0,1.2)
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(b=True, which='major', c='w', lw=2, ls='-')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)
    plt.show()



def plotting_function_model(pred_parameters):

    beta, gamma, sigma, f = pred_parameters
    e0 = 1
    i0 = 0.00
    r0 = 0.00
    d0 = 0.00
    s0 = N - e0 - i0 - r0 -d0
    x0 = [s0, e0, i0, r0, d0]
    t_inf = 5.1

    ret = SEIRD_solver(x0, t, beta, gamma, sigma, f)
    s, e, i, r, d = ret.T
    # print(s, e, i, r)

    # Plot the data on three separate curves for S(t), I(t) and R(t)
    fig = plt.figure(facecolor='w')
    ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
    ax.plot(t, d/N * 100, 'b', alpha=0.5, lw=2, label='Deceased modelled')
    ax.plot(t, deceased/N * 100, 'r', alpha=0.5, lw=2, label='Deceased real')
    # ax.plot(t, r/N * 100, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
    ax.set_xlabel('Time /days')
    ax.set_ylabel('Percentage pop number')
    # ax.set_ylim(0,1.2)
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(b=True, which='major', c='w', lw=2, ls='-')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)
    plt.show()


# print(infective)
# print(recovered)
# print(deceased)

# plotting_function_real(N, t)
# plotting_function_model