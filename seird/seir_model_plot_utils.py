import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.integrate import odeint

import read_data

def covid(x, t):
    s, e, i, r = x 
    dx = np.zeros(4)
    dx[0] = (-beta * s * i) / N
    dx[1] = (beta * s * i) / N - (sigma * e)
    dx[2] = (sigma * e) - (gamma * i)
    dx[3] = gamma * i
    return dx

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

def SEIR_A_model(z, t, r1, r2, beta1, beta2, gamma):
    alpha = 1/7
    S, E, I, R, N = z

    dSdt = -1 * r1 * beta1 * I * S/N - r2 * beta2 * E * S/N
    dEdt = r1 * beta1 * I * S/N + r2 * beta2 * E * S/N - alpha * E
    dIdt = alpha * E - gamma * I
    dRdt = gamma * I
    
    return dSdt, dEdt, dIdt, dRdt, N

def plot_difference(real_data, pred_parameters, N):
    real_I = real_data['totale_positivi'].values 
    real_R = real_data['dimessi_guariti'].values
    real_D = real_data['deceduti'].values 

    t = np.linspace(0, len(real_data['days'])+1, len(real_data['days']))

    beta, sigma, gamma, s0, e0 = pred_parameters
    i0 = real_I[0]
    r0 = real_R[0]

    x0 = [s0, e0, i0, r0]

    x = odeint(SEIR_model, x0, t, args=(beta, sigma, gamma))
    s = x[:, 0]; e = x[:, 1]; i = x[:, 2]; r = x[:, 3]

    # plot the data
    plt.figure(figsize=(8, 5))

    plt.subplot(2, 1, 1)
    plt.title('Real data: ' + 'sigma='+f'{sigma}'+'beta='+f'{beta}'+'gamma='+f'{gamma}')
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


if __name__ == "__main__":
    cases = read_data.get_data_interval('20200301', '20200601', 22)
    data = cases.sort_values('data')
    data['days'] = (data['data'] - data['data'].min()).dt.days

    print(data)

    # real_S = 
    # real_E = 
    real_I = data['totale_positivi'].values 
    real_R = data['dimessi_guariti'].values
    real_D = data['deceduti'].values 

    t = np.linspace(0, len(data['days'])+1, len(data['days']))  # Time frame

    u = 0.2
    t_incubation = 5.1
    t_infective = 3.3
    R0 = 2.4
    N = 542166  # Popolazione P.A. Trento, dati aggiornati al 31.12.2020

    sigma = 1.0e-01
    gamma = 9.0e-02
    beta = 1.4e-01


    e0 = 3.9e+02
    i0 = real_I[0]
    r0 = real_R[0]
    s0 = N - (e0 + i0 + r0)
    x0 = [s0, e0, i0, r0]

    x = odeint(covid, x0, t)
    s = x[:, 0]; e = x[:, 1]; i = x[:, 2]; r = x[:, 3]

    print(real_R)
    print(r)


    # plot the data
    plt.figure(figsize=(8, 5))

    plt.subplot(2, 1, 1)
    plt.title('Real data: ' + 'alpha='+f'{"mona"}'+'beta='+f'{beta}'+'gamma='+f'{gamma}')
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

    # plt.title('Prova')
    # plt.plot(t, (real_I/N)*100, color='blue', lw=3, label='Real I')
    # plt.plot(t, (real_R/N)*100, color='red', lw=3, label='Real R')
    # plt.plot(t, (real_D/N)*100, color='orange', lw=3, label='Real D')
    # plt.ylabel('Fraction')
    # plt.legend()

    plt.show()