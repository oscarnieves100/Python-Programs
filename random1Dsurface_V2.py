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

def random1Dsurface(length=100e-9, corrlength=7e-9, rms_height=20e-9):
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
    
    return xdomain, smoothgaussian, H
    
def EMA(xdomain=0, smoothgaussian=0, H=[0], e1=2.4, e2=3.7, Layers=10,  
        plots=False, test=False):
    
    # Compute areas with layered model
    L = abs(max(xdomain) - min(xdomain))
    dx = abs(xdomain[1] - xdomain[0])
    S = smoothgaussian
    ms = abs(min(S))
    Ms = max(S)
    d1 = Ms # offset above zero 
    d2 = ms # offset below zero
    S = S + ms # shift upwards
    dtot = ms + Ms
    dthickness = dtot/Layers
    layer_volume = L*dthickness
    lines = list( np.linspace(0, dtot, Layers+1) )
    lines.reverse() # top to bottom
    eff = []
    S_slices = []
    V_i = []
    if len(H) <= 1:
        H = S
        
    for nn in range(len(lines)-1):
        S_sliced = []
        for xx in range(len(xdomain)):
            roof = min( lines[nn], S[xx] )
            floor = lines[nn+1]  
            S_sliced.append( max( floor, roof ) )
        
        S_slices.append(S_sliced)
        V_inclusion = dx/2*np.sum(S_sliced[:len(S)-1] + S_sliced[1:]) -\
            lines[nn+1]*L
        V_matrix = layer_volume
        f_i = V_inclusion/V_matrix
        V_i.append(f_i)
        e_m = e1
        e_i = e2
        e_MG = e_m*( e_m + (1+2*f_i)*(e_i-e_m)/3 )/\
                (e_m + (1-f_i)*(e_i - e_m)/3 )
        eff.append( e_MG )
    
    if plots:
        fig, ax = plt.subplots(2,2)
        ax[0,0].plot(xdomain*1e9, np.array(H)*1e9,color='b',linewidth=2)
        ax[0,0].set_xlim([min(xdomain*1e9), max(xdomain*1e9)])
        ax[0,0].set_title("Random surface")
        ax[0,0].set_xlabel("x (nm)")
        ax[0,0].set_ylabel("height (nm)")
        
        for nn in range(len(lines)):
            ax[0,1].axhline(y=lines[nn]*1e9,color='k',linestyle='dashed',
                            linewidth=1.0)
        ax[0,1].plot(xdomain*1e9, np.array(S)*1e9,color='b',linewidth=2)
        ax[0,1].plot(xdomain*1e9, np.array(S_slices[0])*1e9, color='r')
        ax[0,1].set_xlim([min(xdomain*1e9), max(xdomain*1e9)])
        ax[0,1].set_title("Spatially smoothed surface")
        ax[0,1].set_xlabel("x (nm)")
        ax[0,1].set_ylabel("height (nm)")
        
        lines.reverse()
        ax[1,0].plot(np.array(lines[1:])*1e9, eff)
        ax[1,0].scatter(np.array(lines[1:])*1e9, eff, s=20)
        ax[1,0].set_xlabel("depth (nm)")
        ax[1,0].set_ylabel(r"$\varepsilon_{eff}$")
        
        ax[1,1].plot(np.array(lines[1:])*1e9, V_i)
        ax[1,1].scatter(np.array(lines[1:])*1e9, V_i, s=20)
        ax[1,1].set_xlabel("depth (nm)")
        ax[1,1].set_ylabel("Inclusion fraction")
        
        fig.tight_layout()
        plt.savefig("random1Dsurface_test.svg")
        plt.show()
        
    return d1, d2, dthickness, eff

if __name__ == "__main__":
    xdomain, smoothgaussian, H = random1Dsurface()
    d1, d2, dthickness, eff = EMA(xdomain=xdomain, smoothgaussian=smoothgaussian, 
                                  H=H, e1=2.4, e2=3.7, 
                                  plots=True, test=True)