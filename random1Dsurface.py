# -*- coding: utf-8 -*-
"""
1D random spatially-smooth Gaussian surface generator that computes the
effective permittivity of two media meeting at a random surface by using
the Maxwell-Garnett approximation

For relatively smooth surfaces, the length of the structure (e.g. x-axis) must
be at least 10 times the correlation length. When the correlation length
approximates the structural length, you get something close to a sine wave
with half a cyle spanning across the structural length. When the correlation
length is much smaller than the structural length, say by a factor of 100+,
you get a much noisier (less smooth) result. The choice of correlation 
length is up to you, but for realistic surfaces you must consider what
the feature distances between the "bumps" on a real surface are. These
feature distances can be determined experimentally by techniques like AFM, 
but we can in general assume that they are a small fraction of the total 
length of the surface.

The Gaussian smoothing technique is based on the paper:
    "Mrnka, M., 2017, April. Random gaussian rough surfaces for full-wave 
    electromagnetic simulations. In 2017 Conference on Microwave Techniques 
    (COMITE) (pp. 1-4). IEEE."

Here the idea is to "approximate" the effect of surface roughness of
adjoining dielectric media with permittivities e1 and e2 by using
effective medium theory, which is useful for the simulation of certain
electromagnetic structures like Bragg reflectors or multilayer stacks
(e.g. using the Transfer-Matrix Method)

Author: Oscar A. Nieves
"""
import numpy as np 
import matplotlib.pyplot as plt

def random1Dsurface(length=10, corrlength=0.7, rms_height=0.1, 
                     e1=5.8, e2=3.7, plots=False, test=False):
    # Generate spatially smooth Gaussian surface
    L = length
    Lc = corrlength
    factor = 50
    N0 = int( np.ceil( np.log10(factor*L/Lc)/np.log10(2) ) )
    N = 2**N0
    xdomain = np.linspace(-L/2,L/2,N)
    H = rms_height*np.random.normal(0,1,N)
    G = np.exp(-2*(xdomain**2)/Lc**2)
    f = 2*L/np.sqrt(np.pi)/Lc/N
    smoothgaussian = f * np.real( np.fft.ifft( np.fft.fft(H)*np.fft.fft(G) ))
    smoothgaussian = smoothgaussian - np.mean(smoothgaussian)
    rms_smooth = np.sqrt( np.mean(smoothgaussian**2) )
    amplifier = rms_height / rms_smooth
    smoothgaussian = amplifier*smoothgaussian
    rms_smooth = np.sqrt( np.mean(smoothgaussian**2) )
    
    # Compute areas above and below zero
    S = smoothgaussian
    S_up = np.zeros(len(S))
    S_low = np.zeros(len(S))
    for nn in range(len(S)):
        S_up[nn] = max(0, S[nn])
        S_low[nn] = abs( min(0, S[nn]) )
    ms = abs(min(S))
    Ms = max(S)
    
    # Compute volume fractions and effective permittivity
    dx = abs(xdomain[1] - xdomain[0])
    
    V2_up = dx/2*np.sum(S_up[:len(S)-1] + S_up[1:])   
    V1_up = Ms*L - V2_up
    if abs(V1_up/V2_up) <= 1:
        f_i = V1_up/V2_up
        e_m = e2
        e_i = e1
        eff1 = e_m*(2*f_i*(e_i - e_m) + e_i + 2*e_m)/(2*e_m + e_i - f_i*(e_i - e_m))
    else:
        f_i = V2_up/V1_up
        e_m = e1
        e_i = e2
        eff1 = e_m*(2*f_i*(e_i - e_m) + e_i + 2*e_m)/(2*e_m + e_i - f_i*(e_i - e_m))
    if test:
        print("upper volume fraction = " + str(round(f_i,4)))
    
    V2_low = ms*L - dx/2*np.sum(S_low[:len(S)-1] + S_low[1:])
    V1_low = ms*L - V2_low
    if abs(V1_low/V2_low) <= 1:
        f_i = V1_low/V2_low
        e_m = e2
        e_i = e1
        eff2 = e_m*(2*f_i*(e_i - e_m) + e_i + 2*e_m)/(2*e_m + e_i - f_i*(e_i - e_m))
    else:
        f_i = V2_low/V1_low
        e_m = e1
        e_i = e2
        eff2 = e_m*(2*f_i*(e_i - e_m) + e_i + 2*e_m)/(2*e_m + e_i - f_i*(e_i - e_m))
    if test:
        print("lower volume fraction = " + str(round(f_i,4)))
    
    d1 = Ms
    d2 = ms
    
    if test:
        print("effective eps upper = " + str(round(eff1,4)))
        print("effective eps lower = " + str(round(eff2,4)))
    
    if plots:
        fig, ax = plt.subplots(2,1)
        ax[0].plot(xdomain, H,color='b',linewidth=2)
        ax[0].set_xlim([min(xdomain), max(xdomain)])
        ax[0].set_title("Gaussian random surface")
        ax[0].set_xlabel("x (m)")
        ax[0].set_ylabel("height (m)")
        
        ax[1].plot(xdomain, smoothgaussian,color='b',linewidth=2)
        ax[1].axhline(y=0.0,color='r',linestyle='dashed')
        ax[1].axhline(y=Ms,color='k',linestyle='dashed')
        ax[1].axhline(y=-ms,color='k',linestyle='dashed')
        ax[1].set_xlim([min(xdomain), max(xdomain)])
        ax[1].set_title("Spatially smoothed Gaussian surface")
        ax[1].set_xlabel("x (m)")
        ax[1].set_ylabel("height (m)")
        
        fig.tight_layout()
        plt.show()
        
    return d1, d2, eff1, eff2

if __name__ == "__main__":
    random1Dsurface(plots=True, test=True)