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

def SEIRD_solver(t, initial_conditions, params):
    initS, initE, initI, initR, initD, initN = initial_conditions
    alpha, beta, sigma, gamma = params

    return odeint(SEIRD_model, [initS, initE, initI, initR, initD, initN], t, args=(alpha, beta, sigma, gamma))
