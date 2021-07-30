# Arbitrary probability distribution
#
# Generates random numbers according to an arbitrary probability distribution 
# p(x) defined by the user on a domain [a,b]
#
# Author: Oscar A. Nieves
# Last updated: July 30, 2021

import matplotlib.pyplot as plt
import numpy as np 
import math as mt
plt.close('all')
#np.random.seed(0) # Set seed

# Inputs
a = 10
b = 25
x0 = (a+b)/2
samples = 1000
X = np.linspace(a,b,1001)

# Define p(x)
def g(x_var):
    return (x_var - x0)**2
P0 = g(X)

# Normalize p(x)
Psum = sum(P0)
P = P0/Psum

# Cumulative sum p(x)
C = np.cumsum(P)

# Compute numbers
R = np.linspace(a,b,samples)
for n in range(samples):
    index0 = mt.ceil( sum(sum(C[-1]*np.random.rand(1,1) > C)) )
    R[n] = X[index0]
    
# Generate histogram
bins = 50
plt.subplot(1,2,1)
plt.plot(X,P,color='black')
plt.xlim([a,b])
plt.xlabel('X', fontsize=16)
plt.ylabel('p(x)', fontsize=16)

plt.subplot(1,2,2)
plt.hist(R,bins)
plt.xlim([a,b])
plt.xlabel('R', fontsize=16)
plt.ylabel('frequency', fontsize=16)
