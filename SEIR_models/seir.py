from scipy.integrate import odeint
import numpy as np

def SEIR_model(z, t, beta, sigma, gamma):
    """
    Reference https://www.idmod.org/docs/hiv/model-seir.html
    """
    S, E, I, R, N = z
    dSdt = -beta*S*I/N
    dEdt = beta*S*I/N - sigma*E
    dIdt = sigma*E - gamma*I
    dRdt = gamma*I
    return [dSdt, dEdt, dIdt, dRdt, N]

def SEIR_solver(t, initial_conditions, params, infected, recovered):
    initS, initE, initI, initR, initN = initial_conditions
    beta, sigma, gamma = params

    res = odeint(SEIR_model, [initS, initE, initI, initR, initN], t, args=(beta, sigma, gamma))
    _, _, I, R, _ = res.T
    # print(I)
    # COMPUTE THE DIFFERENCE WITH THE ACTUAL DATA
    rmse_I = np.sqrt(np.mean((I - infected) ** 2))
    rmse_R = np.sqrt(np.mean((R - recovered) ** 2))
    return rmse_I, rmse_R