import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from SEIR_models.seir import SEIR_model
from SEIR_models.seird import SEIRD_model

import read_data

def SEIR_A_model(z, t, r1, r2, beta1, beta2, gamma):
    alpha = 1/7
    S, E, I, R, N = z

    dSdt = -1 * r1 * beta1 * I * S/N - r2 * beta2 * E * S/N
    dEdt = r1 * beta1 * I * S/N + r2 * beta2 * E * S/N - alpha * E
    dIdt = alpha * E - gamma * I
    dRdt = gamma * I
    
    return dSdt, dEdt, dIdt, dRdt, N

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

def plot_difference_adv_A(real_data, pred_parameters, N):
    real_I = real_data['totale_positivi'].values 
    real_R = real_data['dimessi_guariti'].values
    real_D = real_data['deceduti'].values 

    t = np.linspace(0, len(real_data['days'])+1, len(real_data['days']))

    r1, r2, beta1, beta2, gamma, s0, e0 = pred_parameters
    i0 = real_I[0]
    r0 = real_R[0]

    x0 = [s0, e0, i0, r0, N]

    x = odeint(SEIR_A_model, x0, t, args=(r1, r2, beta1, beta2, gamma))
    s = x[:, 0]; e = x[:, 1]; i = x[:, 2]; r = x[:, 3]

    # plot the data
    plt.figure(figsize=(8, 5))

    plt.subplot(2, 1, 1)
    plt.title('Real data:')
    plt.plot(t, (real_I/N)*100, color='blue', lw=3, label='Real I')
    plt.plot(t, (i/N)*100, color='red', lw=3, label='Model I')
    plt.ylabel('Fraction')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(t, (real_R/N)*100, color='orange', lw=3, label='Real R')
    plt.plot(t, (r/N)*100, color='purple', lw=3, label='Model R')
    # plt.ylim(0, 0.2)
    plt.xlabel('Time (days)')
    plt.ylabel('Fraction')
    plt.legend()
    plt.show()

def plot_difference_seird(real_data, pred_parameters, N):
    real_I = real_data['totale_positivi'].values 
    real_R = real_data['dimessi_guariti'].values
    real_D = real_data['deceduti'].values 

    t = np.linspace(0, len(real_data['days'])+1, len(real_data['days']))

    alpha, beta, sigma, gamma, e0, i0, r0 = pred_parameters

    d0 = real_D[0]
    s0 = N - e0 - i0 - r0 - d0
    x0 = [s0, e0, i0, r0, d0, N]

    x = odeint(SEIRD_model, x0, t, args=(alpha, beta, sigma, gamma))
    s = x[:, 0]; e = x[:, 1]; i = x[:, 2]; r = x[:, 3]; d = x[:, 4]

    plot_I_R(t, real_I, i, real_R, r)