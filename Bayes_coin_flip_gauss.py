# Bayesian Inference on Biased Coin Flips
#
# Note: requires modules: numpy, pyplot, statistics, celluloid
#
# Author: Oscar A. Nieves
# Last updated: January 19, 2021
import matplotlib.pyplot as plt
import numpy as np 
import statistics as st
from celluloid import Camera
plt.close('all')
np.random.seed(0) # Set seed

# Input parameters
H_bias = 0.35 # Bias of the coin
flips = 1000 # Number of coin flips
N = 200 # number of points in p range
bias_range = np.linspace(0,1,N)

# Prior distribution
# --> Here we use a Gaussian prior distribution
mu = 0.5
sigma = 0.1
X = bias_range
p0 = 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-0.5*((X-mu)/sigma)**2)
p_prior = p0/sum(p0) # normalized on p = [0,1]

# Generate array of coin flips (H = 1, T = 0)
random_array = np.random.uniform(0,1,flips)
flip_array = []
for i in range(flips):
    if random_array[i] <= H_bias: 
        flip_array.append(1)
    else:
        flip_array.append(0)

# Calculate posterior distribution using:
# --> p_posterior = p_prior * p_likelihood / p_evidence
flipV = np.linspace(1,flips,flips)
maxP = np.zeros(flips)
for i in range(flips):
    maxP[i] = np.NaN # make values invisible for plotting
    
yv2 = H_bias * np.ones(flips)
fig = plt.figure()
camera = Camera(fig)

for i in range(flips):
    # Likelihood function (assuming binomial distribution)
    p = bias_range**flip_array[i]
    One = np.ones(len(bias_range))
    q = (One - bias_range)**(1 - flip_array[i])
    p_likelihood = p * q
    p_prod = p_likelihood * p_prior
    p_evidence = sum(p_prod) # normalization factor
    
    # Calculate posterior distribution
    p_posterior = p_prod/p_evidence
    
    # Update prior distribution with posterior
    p_prior = p_posterior 
    
    # Find maximum in p_posterior
    index0 = np.where( p_posterior == np.max(p_posterior) )
    maxP[i] = bias_range[index0[0][0]]
    
    # Generate plot
    plt.subplot(1,2,1)
    plt.plot(bias_range, p_posterior, color='b')
    plt.axvline(x=H_bias,color='r')
    plt.xlabel('theta', fontsize=16)
    plt.ylabel('p_posterior(H)', fontsize=16)
    
    plt.subplot(1,2,2)
    plt.plot(flipV, maxP, color='k')
    plt.plot(flipV, yv2, color='r')
    plt.xlabel('Number of coin flips', fontsize=16)
    plt.ylabel('theta', fontsize=16)
    plt.ylim([0,1])
    camera.snap()
animation = camera.animate()