# Exponential random numbers
#
# Author: Oscar A. Nieves
# Last updated: July 01, 2021
import matplotlib.pyplot as plt
import numpy as np 
plt.close('all')
np.random.seed(0) # Set seed

# Inputs
lambda1 = 0.5
samples = 10000

# Random samples (Uniformly distributed)
Z = np.random.rand(samples,1)

# Exponential random numbers
X = -1/lambda1*np.log(1 - Z)

# Compute mean value
EX = np.mean(X)
EX_ref = 1/lambda1
error_EX = abs(EX_ref - EX)/EX_ref
print(EX)
print(error_EX*100)

# Plot histograms
bins = 50

plt.subplot(1,2,1)
plt.hist(Z,bins)
plt.xlabel('Z ~ Uniform')
plt.ylabel('frequency')

plt.subplot(1,2,2)
plt.hist(X,bins)
plt.xlabel('X ~ Exponential')
plt.ylabel('frequency')