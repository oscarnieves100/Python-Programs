# Least Squares Analysis (LSA) on linear fit
# 
# Author: Oscar A. Nieves
# Last update: January 19 2021
import matplotlib.pyplot as plt
import numpy as np 
import statistics as st
plt.close('all')
np.random.seed(0) # Set seed

# Generate dataset
x = np.linspace(0,10,100) 
Nx = len(x)
noise = np.random.normal(0,1,Nx)
S = 2*x + noise 

# Use LSA to find line of best fit
n = len(S) 
sum_x = sum(x) 
sum_x2 = sum(x**2) 
sum_S = sum(S) 
sum_xS = sum(x*S) 

A = np.array([ [n, sum_x], 
     [sum_x, sum_x2] ])
RHS = np.array([[sum_S], 
                [sum_xS]])
Ainv = np.linalg.inv(A) 
b = np.dot(Ainv,RHS)
b0 = b[0]
b1 = b[1]

# Fit straight line to data and generate plots
y = b0 + b1*x 
print('b0 = ' + str(b0))
print('b1 = ' + str(b1))

# R^2 value
SStot = sum( (S - st.mean(S))**2 )
SSres = sum( (S - y)**2 )
R2 = 1 - SSres/SStot
print('R^2 = ' + str(R2))

# PLOTS
plt.figure(1)
line1 = plt.scatter(x,S,color='r')
line2 = plt.plot(x,y,color='b')
plt.xlabel('x', fontsize=-16)
plt.ylabel('y', fontsize=16)
plt.title("Linear Fit Least Squares")
plt.show()