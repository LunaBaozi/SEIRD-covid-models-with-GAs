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
# t_inf = 5.1
# def SEIRD_model(x, t, f, beta, sigma, gamma):   
#     beta = beta
#     gamma = gamma
#     sigma = sigma
#     f = f
#     S, E, I, R, D, N = x
#     dx = np.zeros(5)
#     dx[0] = -beta * S * I / N
#     dx[1] = beta * S * I / N - sigma * E
#     dx[2] = sigma * E - (1/t_inf) * I
#     dx[3] = ((1-f)/t_inf) * I
#     dx[4] = (f/t_inf) * I
    
    return [dx[0], dx[1], dx[2], dx[3], dx[4], N]

def SEIRD_solver(t, initial_conditions, params):
    initS, initE, initI, initR, initD, initN = initial_conditions
    alpha, beta, sigma, gamma = params

    return odeint(SEIRD_model, [initS, initE, initI, initR, initD, initN], t, args=(alpha, beta, sigma, gamma))
