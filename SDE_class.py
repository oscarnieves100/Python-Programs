# -*- coding: utf-8 -*-
"""
SDE Class

This script contains the SDE class, which is an object representing a single
1D stochastic differential equation with Brownian motion and Poisson jumps of 
the general form:
    
    dX = mu(X,t)dt + sigma(X,t)dW(t) + gamma(X,t)dJ(t)

where X is the stochastic process to be solved for, mu the drift, sigma the
volatility, dW a Wiener step, gamma the jump size and dJ a Poisson step.

Author: Oscar A. Nieves
"""
###############################################################################
# Class definition
###############################################################################
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-poster')

class SDE:
    def __init__(self, mu=lambda X,t: 0.5*X, sigma=lambda X,t: 2*X,
                 gamma=lambda X,t: 0):
        self.mu = mu; self.sigma=sigma; self.gamma=gamma
        self.time = 0; self.solutions = 0
        self.mean = 0; self.variance = 0
        self.skewness = 0; self.kurtosis = 0
    
    # Solve SDE using Euler-Mayurama scheme
    def solve(self, MC=10, Nt=100, T=1, X0=1, all_solutions=True, set_seed=0):
        np.random.seed(set_seed) # set random seed
        t = np.linspace(0,T,Nt)
        dt = abs(t[1]-t[0])
        X = np.zeros((MC,Nt))
        X[:,0] = X0
        dW = np.sqrt(dt)*np.random.standard_normal(size=(MC,Nt))
        dJ = np.random.poisson(lam=dt, size=(MC,Nt))
        for n in range(Nt-1):
            X[:,n+1] = X[:,n] + self.mu(X[:,n],t[n])*dt + \
                self.sigma(X[:,n],t[n])*dW[:,n] + self.gamma(X[:,n],t[n])*dJ[:,n]
        self.time = t
        if all_solutions:
            self.solutions = X
        else: 
            self.solutions = X[0,:]
        
        # compute statistics
        self.mean = np.mean(X, axis=0)
        self.variance = np.mean(X**2, axis=0) - self.mean**2
        stdev = np.sqrt(self.variance)
        self.skewness = np.mean( ((X - self.mean[None,:])/stdev[None,:])**3, axis=0)
        self.kurtosis = np.mean( ((X - self.mean[None,:])/stdev[None,:])**4, axis=0)
        
        # Adjust values at t=0
        self.skewness[0], self.kurtosis[0] = 0, 0
        
    # Plot solutions
    # mode = "trajectories" plots Monte Carlo runs
    # mode = "mean", "variance", "skewness", "kurtosis" plots everything else
    def plot_solution(self, mode="trajectories", title0=" ", font_size=20):
        try:
            MC, Nt = np.shape(self.solutions)
            
            fig, ax = plt.subplots(figsize=(6,6), dpi=600) 
            fontS = font_size
            plt.xlim([min(self.time), max(self.time)])
            plt.xlabel(r"$t\ [s]$", fontsize=fontS)
            plt.grid(visible=False)
            
            if mode=="trajectories":
                for m in range(MC):
                    plt.plot(self.time, self.solutions[m,:], 'o', linestyle="solid",
                             linewidth=1, markersize=4)
                plt.ylabel(r"$X(t)$", fontsize=fontS)
                plt.ylim([np.min(self.solutions), np.max(self.solutions)])
                
            elif mode=="mean":
                plt.plot(self.time, self.mean, 'o', linestyle="solid",
                         linewidth=2.5, markersize=4, color="black")
                plt.ylabel(r"$\mathbb{E}[X(t)]$", fontsize=fontS)
                plt.ylim([np.min(self.mean), np.max(self.mean)])
                
            elif mode=="variance":
                plt.plot(self.time, self.variance, 'o', linestyle="solid",
                         linewidth=2.5, markersize=4, color="black")
                plt.ylabel(r"$Var[X(t)]$", fontsize=fontS)
                plt.ylim([np.min(self.variance), np.max(self.variance)])                
                
            elif mode=="skewness":
                plt.plot(self.time, self.skewness, 'o', linestyle="solid",
                         linewidth=2.5, markersize=4, color="black")
                plt.ylabel(r"$Skew[X(t)]$", fontsize=fontS)
                plt.ylim([np.min(self.skewness), np.max(self.skewness)])
                
            elif mode=="kurtosis":
                plt.plot(self.time, self.kurtosis, 'o', linestyle="solid",
                         linewidth=2.5, markersize=4, color="black")
                plt.ylabel(r"$Kurt[X(t)]$", fontsize=fontS)
                plt.ylim([np.min(self.kurtosis), np.max(self.kurtosis)])
            
            plt.title(title0)
            plt.tight_layout()
            plt.show()
            
        except:
            raise("Error plotting solution. Ensure solutions are in the right format.")
    
    # Plots all results in sequence
    def plot_all(self, title0=" ", font_size=20):
        self.plot_solution(mode="trajectories", title0=title0, font_size=font_size)
        self.plot_solution(mode="mean", title0=title0, font_size=font_size)
        self.plot_solution(mode="variance", title0=title0, font_size=font_size)
        self.plot_solution(mode="skewness", title0=title0, font_size=font_size)
        self.plot_solution(mode="kurtosis", title0=title0, font_size=font_size)
        
###############################################################################
# Test Functions
###############################################################################
if __name__ == "__main__":    
    # Inputs
    title0 = r"$dX(t) = X(t)dt + 0.25 X(t) dW(t) + 0.1 X(t) dJ(t)$"
    mu = lambda X,t: X
    sigma = lambda X,t: 0.25*X
    gamma = lambda X,t: 0.1*X
    eq = SDE(mu=mu, sigma=sigma, gamma=gamma)
    
    # Solve equation
    eq.solve(MC=20, Nt=101, T=2)
    
    # Plot results
    eq.plot_all(title0=title0, font_size=20)