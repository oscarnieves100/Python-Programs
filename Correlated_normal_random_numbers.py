# Correlated normal random numbers
#
# Author: Oscar A. Nieves
# Last updated: July 01, 2021

import matplotlib.pyplot as plt
import numpy as np 
plt.close('all')
np.random.seed(0) # Set seed

# Inputs
samples = 1000

# Random samples (Uniformly distributed)
U1 = np.random.rand(samples,1)
U2 = np.random.rand(samples,1)

# Random samples (normally distributed uncorrelated) 
S1 = np.sqrt(-2*np.log(U1))*np.cos(2*np.pi*U2)
S2 = np.sqrt(-2*np.log(U1))*np.sin(2*np.pi*U2)

E_S1 = np.mean(S1)
Var_S1 = np.mean(S1**2) - E_S1**2
sigma_S1 = np.sqrt(Var_S1)
E_S2 = np.mean(S2)
Var_S2 = np.mean(S2**2) - E_S2**2
sigma_S2 = np.sqrt(Var_S2)

Cov_S1_S2 = np.mean(S1*S2) - E_S1*E_S2
Corr_S1_S2 = Cov_S1_S2/sigma_S1/sigma_S2

print('corr(S1,S2) = ' + str(Corr_S1_S2))

# Correlated random samples
mu_x = 0.5
mu_y = 0.66
sigma_x = 0.85
sigma_y = 1.24
rho = 0.5
X = mu_x + sigma_x * S1
Y = mu_y + sigma_y * (rho*S1 + np.sqrt(1-rho**2)*S2)

E_X = np.mean(X)
Var_X = np.mean(X**2) - E_X**2
sigma_X = np.sqrt(Var_X)
E_Y = np.mean(Y)
Var_Y = np.mean(Y**2) - E_Y**2
sigma_Y = np.sqrt(Var_Y)

Cov_X_Y = np.mean(X*Y) - E_X*E_Y
Corr_X_Y = Cov_X_Y/sigma_X/sigma_Y

print('corr(X,Y) = ' + str(Corr_X_Y))

# Generate plots
plt.subplot(1,2,1)
plt.plot(S1,S2,linestyle="",marker="o",color="blue")
plt.xlabel('S1', fontsize=16)
plt.ylabel('S2', fontsize=16)

plt.subplot(1,2,2)
plt.plot(X,Y,linestyle="",marker="o",color="green")
plt.xlabel('X', fontsize=16)
plt.ylabel('Y', fontsize=16)