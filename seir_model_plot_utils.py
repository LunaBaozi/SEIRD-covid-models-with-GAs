import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from SEIR_models.seir import SEIR_model
from SEIR_models.seird import SEIRD_model

def plot_I_R(t, real_I, pred_I, real_R, pred_R):
    plt.figure(figsize=(8, 5))

    plt.subplot(2, 1, 1)
    plt.title('Real data:')
    plt.plot(t, real_I, color='blue', lw=3, label='Real I')
    plt.plot(t, pred_I, color='red', lw=3, label='Model I')
    plt.ylabel('Infected')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(t, real_R, color='orange', lw=3, label='Real R')
    plt.plot(t, pred_R, color='purple', lw=3, label='Model R')
    plt.xlabel('Time (days)')
    plt.ylabel('Recovered')
    plt.legend()
    plt.show()

def plot_I_R_D(t, real_I, pred_I, real_R, pred_R, real_D=None, pred_D=None):
    if real_D is not None and pred_D is not None:
        plt.figure(figsize=(8, 7))

        plt.subplot(3, 1, 1)
        plt.title('Real data:')
        plt.plot(t, real_I, color='blue', lw=3, label='Real I')
        plt.plot(t, pred_I, color='red', lw=3, label='Model I')
        plt.ylabel('Infected')
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.plot(t, real_R, color='orange', lw=3, label='Real R')
        plt.plot(t, pred_R, color='purple', lw=3, label='Model R')
        plt.ylabel('Recovered')
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.plot(t, real_D, color='black', lw=3, label='Real D')
        plt.plot(t, pred_D, color='gray', lw=3, label='Model D')
        plt.xlabel('Time (days)')
        plt.ylabel('Death')
        plt.legend()
        plt.show()
    else:
        plot_I_R(t, real_I, pred_I, real_R, pred_R)


def plot_difference(real_data, pred_parameters, N):
    real_I = real_data['totale_positivi'].values 
    real_R = real_data['dimessi_guariti'].values
    real_D = real_data['deceduti'].values 

    t = np.linspace(0, len(real_data['days'])+1, len(real_data['days']))

    beta, sigma, gamma, e0, i0, r0 = pred_parameters
    s0 = N - e0 - i0 - r0
    x0 = [s0, e0, i0, r0, N]

    x = odeint(SEIR_model, x0, t, args=(beta, sigma, gamma))
    s = x[:, 0]; e = x[:, 1]; i = x[:, 2]; r = x[:, 3]

    plot_I_R(t, real_I, i, real_R, r)

def plot_difference_seird(real_data, pred_parameters, N):
    real_I = real_data['totale_positivi'].values 
    real_R = real_data['dimessi_guariti'].values
    real_D = real_data['deceduti'].values 
    tot_pos = real_data['totale_positivi'].values
    tot_rec = real_data['dimessi_guariti'].values
    tot_dec = real_data['deceduti'].values

    t = real_data['days'].values  

    alpha, beta, sigma, gamma, e0 = pred_parameters

    s0 = N - e0 - tot_pos[0] - tot_rec[0] - tot_dec[0]
    x0 = [s0, e0, tot_pos[0], tot_rec[0], tot_dec[0], N]

    x = odeint(SEIRD_model, x0, t, args=(alpha, beta, sigma, gamma))
    s = x[:, 0]; e = x[:, 1]; i = x[:, 2]; r = x[:, 3]; d = x[:, 4]

    plot_I_R_D(t, real_I, i, real_R, r, real_D, d)