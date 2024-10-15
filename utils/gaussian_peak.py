from scipy.signal import find_peaks
import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import curve_fit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, find_peaks_cwt
import os

# Helper function to fit the peaks
def gauss(x, *params):
    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        a = params[i]
        b = params[i+1]
        c = params[i+2]
        y += a * np.exp(-(x - b)**2 / (2*c**2))
    return y

folder_name = "C:/Users/kvriz/Desktop/Polybot_ECPs\datasets/absorption_spectra/literature_data/ACS Macro Lett., 5, 714-717, 2016"
file_name = "PProDOT_new.csv"
filename = os.path.join(folder_name, file_name)
data = pd.read_csv(filename)
# Detecting peaks with a prominence that helps in distinguishing peaks
peaks, _ = find_peaks(data['Absorbance'], prominence=0.1) 

# Estimating initial parameters for the Gaussian fits:
# Amplitude is the peak intensity.
# Mean is the x value (energy) at the peak.
# Standard deviation (std) is related to the width of the peak.
initial_params = []
for peak in peaks:
    amplitude = data['Absorbance'][peak]
    mean = data['Wavelength (nm)'][peak]
    std = 2 # Starting with an arbitrary value, to be optimized
    initial_params.extend([amplitude, mean, std])

# Using curve_fit to fit the data to the gauss function
params, _ = curve_fit(gauss, data['Wavelength (nm)'], data['Absorbance'], p0=initial_params)

# Using the optimized parameters to plot the fitted Gaussian peaks and the reconstructed spectrum
fitted_peaks = gauss(data['Wavelength (nm)'], *params)
reconstructed_spectrum = np.sum([gauss(data['Wavelength (nm)'], *params[i:i+3]) for i in range(0, len(params), 3)], axis=0)

# Plotting the original data, the fitted peaks, and the reconstructed spectrum
plt.figure(figsize=(10, 6))
plt.plot(data['Wavelength (nm)'], data['Absorbance'], label='Original Spectrum')
plt.plot(data['Wavelength (nm)'], reconstructed_spectrum, label='Reconstructed Spectrum', linestyle='--')
plt.plot(data['Wavelength (nm)'], fitted_peaks, label='Fitted Peaks', linestyle='-.')
for i in range(0, len(params), 3):
    plt.fill_between(data['Wavelength (nm)'], gauss(data['Wavelength (nm)'], *params[i:i+3]), alpha=0.3)
plt.xlabel('Energy (eV)')
plt.ylabel('Intensity (a.u.)')
plt.title('Peak Decomposition and Reconstruction')
plt.legend()
plt.grid(True)
plt.show()

# Plotting the original data and the fitted Gaussian peaks
plt.figure(figsize=(10, 6))
plt.plot(data['Wavelength (nm)'], data['Absorbance'], label='Original Spectrum', zorder=1)

# Plotting each fitted Gaussian peak
for i in range(0, len(params), 3):
    peak_curve = gauss(data['Absorbance'], *params[i:i+3])
    plt.plot(data['Wavelength (nm)'], peak_curve, linestyle='--', zorder=2)

# Plotting the reconstructed spectrum (sum of fitted Gaussian peaks)
plt.plot(data['Wavelength (nm)'], reconstructed_spectrum, label='Reconstructed Spectrum', color='red', zorder=3)

plt.xlabel('Energy (eV)')
plt.ylabel('Intensity (a.u.)')
plt.title('Peak Decomposition and Reconstruction')
plt.legend()
plt.grid(True)
plt.show()