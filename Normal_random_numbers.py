# Normal random numbers
#
# Author: Oscar A. Nieves
# Last updated: July 01, 2021
import matplotlib.pyplot as plt
import numpy as np 
plt.close('all')
np.random.seed(0) # Set seed

# Inputs
mu = 0.5
sigma = 0.75
samples = 100000

# Random samples (Uniformly distributed)
A = np.random.rand(samples,1)
B = np.random.rand(samples,1)

# Normal random numbers N(mu, sigma^2)
X = mu + sigma * np.sqrt(-2*np.log(A))*np.cos(2*np.pi*B)
Y = mu + sigma * np.sqrt(-2*np.log(A))*np.sin(2*np.pi*B)

# Compute statistics
EX = round(np.mean(X),2)
VarX = round(np.var(X),2)
EY = round(np.mean(Y),2)
VarY = round(np.var(Y),2)

# Plot histograms
bins = 50

plt.subplot(1,2,1)
plt.hist(X,bins)
plt.xlabel('X ~ N(mu = ' + str(EX) + ', sigma^2 = ' + str(VarX) + ')', fontsize=16)
plt.ylabel('frequency', fontsize=16)

plt.subplot(1,2,2)
plt.hist(Y,bins)
plt.xlabel('Y ~ N(mu = ' + str(EY) + ', sigma^2 = ' + str(VarY) + ')', fontsize=16 )
plt.ylabel('frequency', fontsize=16)