# Least Squares Analysis (LSA) on arbitrary polynomial fit
# 
# Author: Oscar A. Nieves
# Last update: January 19 2021
import matplotlib.pyplot as plt
import numpy as np 
import statistics as st
plt.close('all')
np.random.seed(0) # Set seed

# Generate dataset
N = 3 # order of the polynomial fit (predicted)
x = np.linspace(-3,3,100) 
Nx = len(x)
noise = np.random.normal(0,1,Nx)
S = 4*x**3 - 3*x**2 + 6*x + 10*noise 

# Use LSA to find line of best fit
n = len(S)
A = np.zeros((N+1,N+1))
RHS = np.zeros((N+1,1))
for j in range(N+1):
    powers = np.linspace(j,N+j,N+1)
    for i in range(len(powers)):
        A[j,i] = sum(x**powers[i])
    RHS[j,0] = sum(S*x**j)

Ainv = np.linalg.inv(A) 
b = np.dot(Ainv,RHS)
print(b)

# Fit straight line to data and generate plots
Y0 = np.zeros((N+1,n))
for j in range(N+1):
    Y0[j,:] = b[j] * x**j
y = Y0.sum(axis=0)

# R^2 value
SStot = sum( (S - st.mean(S))**2 )
SSres = sum( (S - y)**2 )
R2 = 1 - SSres/SStot
print('R^2 = ' + str(R2))

# PLOTS
plt.figure(1)
plt.scatter(x,S,color='r',label='Data')
plt.plot(x,y,color='b',label='Fit')
plt.xlabel('x', fontsize=16)
plt.ylabel('y', fontsize=16)
plt.title("Polynomial Fit Least Squares")
plt.show()
plt.legend()
