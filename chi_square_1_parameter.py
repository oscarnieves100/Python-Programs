# Chi-square Test for Goodness of Fit using Single Parameter
# (Only the slope of a straight-line of best fit for a given set of data)
# 
# Author: Oscar A. Nieves
# Last update: January 19 2021
import matplotlib.pyplot as plt
import numpy as np 
plt.close('all')

# Generate Input Data
x = np.array([0.1, 0.6, 1.1, 1.6, 2.1, 2.6, 3.1, 3.6, 4.1,
5.6, 6.1, 6.6, 7.1, 7.6, 8.1, 8.6, 9.1])

y = np.array([34.1329, 98.7892, 121.0725, 180.3328,
260.5684, 320.9553, 380.3028, 407.3759, 453.7503,
576.9329,602.0845, 699.0915, 771.2271, 796.6707,
877.0763, 915.3649, 1000.7312])

yerr = np.array([5.8423, 9.9393, 11.0033, 13.4288, 15.3743,
17.9152, 19.5014, 20.1836, 21.3014, 22.516, 24.0194,
24.5374, 26.4403, 27.771, 28.2254, 28.0686, 29.6155])

# Calculate chi-square parameters
s_yy = sum( y**2/yerr**2 )
s_xx = sum( x**2/yerr**2 )
s_xy = sum( (y*x)/yerr**2 )
A = s_xy/s_xx # slope of best-fit line
sigma_A = 1/np.sqrt(s_xx)
minchi2 = s_yy - s_xy**2/s_xx

# Plot interval for chi-square parameter
twosigma = np.array([A-2*sigma_A, A+2*sigma_A])

# Create parameter range for slope (line of best fit)
a = np.linspace(twosigma[0], twosigma[1],1000)

# Calculate chi-square over parameter grid
chi2 = s_yy + (a**2)*s_xx - 2*a*s_xy;

# Generate Plot with Error Bars
plt.figure(1)
plt.plot(x,A*x)
plt.errorbar(x,y, yerr, linestyle='None' ,fmt=' .k')
plt.xlabel('x', fontsize=-16)
plt.ylabel('y', fontsize=16)
plt.grid(True)
plt.title("y vs x data with y-error bars")

# Display chi-square vs. slope
plt.figure(2)
plt.plot (a,chi2,color='b')
plt.axhline (y=minchi2+1, color='r')
plt.xlabel('slope' , fontsize=16)
plt.ylabel('chisq' , fontsize=16)
plt.grid(True)
plt.title ("Chi-square as a function of slope \n %4d points \
chisg min \ =%6.2f best slope =%7.2f " %(x.size,minchi2,A))