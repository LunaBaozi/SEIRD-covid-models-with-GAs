import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.integrate import odeint

import read_data

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

sigma = 1/t_incubation
gamma = 1/t_infective
beta = R0*gamma




e0 = 1/N
i0 = 0.00
r0 = 0.00
s0 = 1 - e0 - i0 - r0 
x0 = [s0, e0, i0, r0]

def covid(x, t):
    s, e, i, r = x 
    dx = np.zeros(4)
    dx[0] = (-beta * s * i) / N
    dx[1] = (beta * s * i) / N - (sigma * e)
    dx[2] = (sigma * e) - (gamma * i)
    dx[3] = gamma * i
    return dx

x = odeint(covid, x0, t)
s = x[:, 0]; e = x[:, 1]; i = x[:, 2]; r = x[:, 3]

print(real_R)
print(r)


# plot the data
plt.figure(figsize=(8, 5))

plt.subplot(2, 1, 1)
plt.title('Real data: ' + 'alpha='+f'{alpha}'+'beta='+f'{beta}'+'gamma='+f'{gamma}')
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