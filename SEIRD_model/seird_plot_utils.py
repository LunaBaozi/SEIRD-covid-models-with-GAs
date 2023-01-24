import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime, timedelta
from matplotlib.dates import DateFormatter
from matplotlib import pyplot as plt
from matplotlib import dates as mdates

import read_data
from seird_problem import SEIRD_solver

data = read_data.get_data_interval('20200301', '20200401', 22)

myDates = [list(str(x).split(' '))[0] for x in data['data']]
infective = data['totale_positivi'].values 
recovered = data['dimessi_guariti'].values
deceased = data['deceduti'].values 

N = 542166
num_vars = 4

t = np.linspace(0, len(data), len(data))
print(len(t))
N = 542166


def plotting_function_model(pred_parameters):
    """
    Plotting function predicted data vs real data.
    :param pred_parameters:     dataframe of predicted values for SEIRD parameters
    """
    beta, gamma, sigma, f = pred_parameters['beta'], pred_parameters['gamma'], pred_parameters['sigma'], pred_parameters['f']
    beta, gamma, sigma, f = beta.to_numpy(), gamma.to_numpy(), sigma.to_numpy(), f.to_numpy()
    e0 = 1
    i0 = infective[0]
    r0 = recovered[0]
    d0 = deceased[0]
    s0 = N - e0 - i0 - r0 -d0
    x0 = [s0, e0, i0, r0, d0]
    t_inf = 5.1

    solutions = []
    death = []

    for i in range(len(beta)):
        ret = SEIRD_solver(x0, t, beta[i], gamma[i], sigma[i], f[i])
        s, e, i, r, d = ret.T
        death.append(d)
        solutions.append([s,e,i,r,d])

    df=pd.DataFrame(solutions, columns=['s', 'e', 'i', 'r', 'd']) 
    fig = plt.figure(facecolor='w')
    ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)

    for i in death:
        ax.plot(myDates, i/N * 100, 'b', alpha=0.5, lw=2)

    ax.plot(myDates, i/N * 100, 'b', alpha=0.5, lw=2, label='Deceased modelled')
    ax.plot(myDates, (data['deceduti'].values)/N * 100, 'r', alpha=0.5, lw=2, label='Deceased real')
    ax.set_xlabel('Date (days)')
    ax.set_ylabel('Percentage population number')
    for label in ax.get_xticklabels(which='major'):
        label.set(rotation=30, horizontalalignment='right')
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(b=True, which='major', c='w', lw=2, ls='-')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)
    plt.show()
