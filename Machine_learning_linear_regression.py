# Linear Regression of a straight-line using both Least-Squares Analysis 
# (i.e. linear algebra methods) and a simple Gradient descent machine learning
# algorithm in which the linear parameters in y = b0 + b1*x get updated
# through several epochs until a certain convergence is achieved
# 
# Author: Oscar A. Nieves
# Last update: January 19 2021
import matplotlib.pyplot as plt
import numpy as np 
import statistics as st
plt.close('all')
np.random.seed(0) # Set seed

# Generate noisy dataset
x = np.linspace(0,10,100) 
Nx = len(x)
noise = np.random.normal(0,1,Nx)
S = 2*x + noise 

## --- Use LSA to find line of best fit --- ##
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
print("LSA model:")
y = b0 + b1*x 
print('b0 = ' + str(b0[0]))
print('b1 = ' + str(b1[0]))

# R^2 value from LSA
SStot = sum( (S - st.mean(S))**2 )
SSres = sum( (S - y)**2 )
R2 = 1 - SSres/SStot
print('R^2 = ' + str(R2))

## --- Machine Learning model (using Gradient Descent) --- ##
# Define learning model as follows: let L be the mean squared error
# L = 1/N*sum( (S - (b0 + b1*x))^2 )
# Then we calculate parameters as follows:
# b0_i <-- a * -2/N*(S_i - (b0_{i-1} + b1_{i-1}*x_i ))
# b1_i <-- a * -2/N*x_i*(S_i - (b0_{i-1} + b1_{i-1}*x_i ))
# where a is an arbitrary learning rate (e.g. 0.05)

# Updating function
def update_values(xdata,ydata,b0_ML,b1_ML,a):
    dL_db0 = 0.0
    dL_db1 = 0.0 # initial guesses
    NL = len(xdata)
    
    # Loop over values in xdata
    for i in range(NL):
        dL_db1 += -2*xdata[i]*(ydata[i] - (b0_ML + b1_ML*xdata[i]))
        dL_db0 += -2*(ydata[i] - (b0_ML + b1_ML*xdata[i]))
        
    # Update values b0_ML, b1_ML
    b0_ML = b0_ML - (1/float(NL))*dL_db0*a
    b1_ML = b1_ML - (1/float(NL))*dL_db1*a
    return b0_ML, b1_ML

# Average loss function L
def avg_loss(xdata,ydata,b0_ML,b1_ML):
    NL = len(xdata)
    tot_err = 0.0
    for i in range(NL):
        tot_err += (ydata[i] - (b0_ML + b1_ML*xdata[i]))
    return tot_err/float(NL)

# Training function
def train(xdata,ydata,b0_ML,b1_ML,a,epochs):
    for j in range(epochs):
        b0_ML, b1_ML = update_values(xdata,ydata,b0_ML,b1_ML,a)
    
    # Print progress
    if j % 1000 == 0: # print every 1000 epochs
        print("epoch:", j, "loss: ", avg_loss(xdata,ydata,b0_ML,b1_ML))
    
    return b0_ML, b1_ML

# Run train session
a = 0.001
b0_ML = 0.0
b1_ML = 0.0
epochs = 10000
b0_ML, b1_ML = train(x,S,b0_ML,b1_ML,a,epochs)
y_ML = b0_ML + b1_ML*x
SStot_ML = sum( (S - st.mean(S))**2 )
SSres_ML = sum( (S - y_ML)**2 )
R2_ML = 1 - SSres_ML/SStot_ML
print(' ')
print("Machine learning model:")
print('b0 = ' + str(b0_ML))
print('b1 = ' + str(b1_ML))
print('R^2 = ' + str(R2_ML))

## --- PLOTS and comparison --- ##
plt.figure(1)
plt.subplots_adjust(wspace=0.5)

# LSA model
plt.subplot(1,2,1)
plt.scatter(x,S,color='r',label='Data')
plt.plot(x,y,color='b',label='Fit')
plt.xlabel('x', fontsize=16)
plt.ylabel('y', fontsize=16)
plt.title("Linear Fit Least Squares")
plt.show()
plt.legend()

# Machine learning model
plt.subplot(1,2,2)
plt.scatter(x,S,color='r',label='Data')
plt.plot(x,y_ML,color='b',label='Fit')
plt.xlabel('x', fontsize=16)
plt.ylabel('y', fontsize=16)
plt.title("Machine Learning Model")
plt.show()
plt.legend()