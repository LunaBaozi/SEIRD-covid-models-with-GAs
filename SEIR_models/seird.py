from scipy.integrate import odeint
import numpy as np

def SEIRD_model(z, t, alpha, beta, sigma, gamma):
    S, E, I, R, D, N = z

    dSdt = -beta * S * I/N
    dEdt = beta * S * I/N - sigma * E
    dIdt = sigma * E - (gamma + alpha) * I
    dRdt = gamma * I
    dDdt = alpha * I

    return [dSdt, dEdt, dIdt, dRdt, dDdt, N]

def SEIRD_solver(t, initial_conditions, params, infected, recovered, death):
    initS, initE, initI, initR, initD, initN = initial_conditions
    alpha, beta, sigma, gamma = params

    res = odeint(SEIRD_model, [initS, initE, initI, initR, initD, initN], t, args=(alpha, beta, sigma, gamma))
    _, _, I, R, D, _ = res.T

    rmse_I = np.sqrt(np.mean((I - infected) ** 2))
    rmse_R = np.sqrt(np.mean((R - recovered) ** 2))
    rmse_D = np.sqrt(np.mean((D - death) ** 2))
    return rmse_I, rmse_R, rmse_D