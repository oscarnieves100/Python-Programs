# -*- coding: utf-8 -*-
"""
Stochastic simulation of the velocity of a Brownian particle in 1D travelling
through a viscous medium. Based on the medium article:

https://oscarnieves100.medium.com/the-fluctuation-dissipation-theorem-793aec4608fd

Author: Oscar A. Nieves
"""
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-poster')
np.random.seed(1)

# Input parameters
v0 = 10 # initial velocity (m/s)
Q = 0.25 # Noise strength
zeta = 0.5 # zeta = gamma/m
t_tot = 20.0 # total simulation time (s)
MonteCarlo = 1000 # Monte Carlo simulations

# Create arrays
t = np.linspace(0, t_tot, 101) # time array
dt = t[1]-t[0] # time step-size (s)
Nt = len(t)
f = np.sqrt(Q/dt) * np.random.standard_normal(size=(MonteCarlo, Nt))

v = np.zeros( (MonteCarlo, Nt) )
v[:,0] = v0 # insert initial condition

# Analytic solutions
v_mean = v0*np.exp(-zeta*t) 
v_var = Q/2/zeta*(1 - np.exp(-2*zeta*t))

# 95% confidence intervals (analytic)
v_CI_upper = v_mean + 1.96*np.sqrt(v_var)
v_CI_lower = v_mean - 1.96*np.sqrt(v_var)

# Euler-Mayurama numerical solver
for MC in range(MonteCarlo):
    for n in range(Nt-1):
        v[:,n+1] = (1 - zeta*dt)*v[:,n] + dt*f[:,n]

v_mean_num = np.mean(v, axis=0)
v_var_num = np.mean(v**2, axis=0) - v_mean_num**2
        
# Plot results
plt.figure(figsize=(6,6), dpi=1000)
plt.plot(t, v_CI_upper, linestyle="dashed", color='red', label="95% upper CI")
plt.plot(t, v_CI_lower, linestyle="dashed", color='darkgreen', label="95% lower CI")
plt.plot(t, v_mean, linestyle="solid", color='blue', label="analytic mean")
plt.plot(t, v_mean_num, linestyle="dashdot", color='black', label="numeric mean")
# for MC in range(MonteCarlo):
#     plt.plot(t, v[MC,:], 'o', linestyle="solid", linewidth=1, markersize=4)

plt.xlabel(r"$t\ [sec]$", fontsize=20)
plt.ylabel(r"$v(t)\ [m/s]$", fontsize=20)
plt.legend(frameon=False, loc="upper right", fontsize=14)
plt.grid(visible=False)
plt.tight_layout()
plt.show()