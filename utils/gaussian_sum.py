from scipy.optimize import least_squares
from scipy.optimize import curve_fit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, find_peaks_cwt
import os


def gauss(x, *params):
    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        a = params[i]
        b = params[i+1]
        c = params[i+2]
        y += a * np.exp(-(x - b)**2 / (2*c**2))
    return y

# Define a function that models a sum of 20 Gaussian peaks
def gauss_sum_model(x, params):
    return np.sum([gauss(x, params[i], params[i + 1], params[i + 2]) for i in range(0, len(params), 3)], axis=0)

# The total number of peaks we want to fit
num_peaks = 20

folder_name = "C:/Users/kvriz/Desktop/Polybot_ECPs\datasets/absorption_spectra/literature_data/ACS Macro Lett., 5, 714-717, 2016"
file_name = "PProDOT_new.csv"
filename = os.path.join(folder_name, file_name)
data = pd.read_csv(filename)
# Generate an initial guess for the parameters of each Gaussian peak
# We'll use evenly spaced means, and guess that each peak has the same width and height
energy_span = data['Wavelength (nm)'].max() - data['Wavelength (nm)'].min()
peak_spacing = energy_span / (num_peaks + 1)
initial_means = np.linspace(data['Wavelength (nm)'].min() + peak_spacing, data['Wavelength (nm)'].max() - peak_spacing, num_peaks)
initial_heights = np.full(num_peaks, 0.5 * data['Absorbance'].max())
initial_widths = np.full(num_peaks, peak_spacing / 2)

# # Flatten the initial parameters into a single array (for the optimizer)
initial_params = np.ravel(list(zip(initial_heights, initial_means, initial_widths)))

# # Define the objective function for the least squares optimizer
def objective_function(params, x, y):
    model_y = gauss_sum_model(x, params)
    return model_y - y

# # Perform the optimization using least squares
# result = least_squares(objective_function, initial_params, args=(data['Wavelength (nm)'], data['Absorbance']))

# # Extract the optimized parameters
# optimized_params = result.x

# Plot the results
# plt.figure(figsize=(10, 6))
# plt.plot(data['Wavelength (nm)'], data['Absorbance'], label='Original Spectrum', zorder=1)
# plt.plot(data['Wavelength (nm)'], gauss_sum_model(data['Wavelength (nm)'], optimized_params), label='Fitted 20 Peaks', color='red', zorder=2)
# plt.xlabel('Energy (eV)')
# plt.ylabel('Absorbance (a.u.)')
# plt.title('Multi-Peak Fitting with 20 Gaussian Peaks')
# plt.legend()
# plt.grid(True)
# plt.show()from scipy.optimize import least_squares


# Ensure the initial parameters are within the bounds
# Set initial amplitudes to a fraction of the max Absorbance to ensure they are positive
initial_amplitudes = np.full(num_peaks, 0.1 * data['Absorbance'].max())

# Centers should be within the data range, we distribute them evenly across the range
initial_centers = np.linspace(data['Wavelength (nm)'].min() + 2.5, data['Wavelength (nm)'].max() - 2.5, num_peaks)

# Set initial widths (FWHM) to a reasonable guess within the bounds
initial_widths = np.full(num_peaks, 3)  # Halfway between 0 and 6 eV

# Flatten the initial parameters into a single array (for the optimizer)
initial_params = np.ravel(list(zip(initial_amplitudes, initial_centers, initial_widths)))

# Create bounds for each parameter
lower_bounds = [0, data['Wavelength (nm)'].min(), 0] * num_peaks
upper_bounds = [np.inf, data['Wavelength (nm)'].max(), 6] * num_peaks
for i in range(num_peaks):
    lower_bounds[i * 3 + 1] = initial_centers[i] - 2.5
    upper_bounds[i * 3 + 1] = initial_centers[i] + 2.5

# Perform the optimization with bounds again
result_with_bounds = least_squares(objective_function, initial_params, args=(data['Wavelength (nm)'], data['Absorbance']), bounds=(lower_bounds, upper_bounds))

# Check if optimization was successful
if result_with_bounds.success:
    # Extract the optimized parameters with bounds
    optimized_params_with_bounds = result_with_bounds.x

    # Plot the results with bounds
    plt.figure(figsize=(10, 6))
    plt.plot(data['Wavelength (nm)'], data['Absorbance'], label='Original Spectrum', zorder=1)
    plt.plot(data['Wavelength (nm)'], gauss_sum_model(data['Wavelength (nm)'], optimized_params_with_bounds), label='Fitted 20 Peaks with Bounds', color='red', zorder=2)
    plt.xlabel('Energy (eV)')
    plt.ylabel('Absorbance (a.u.)')
    plt.title('Multi-Peak Fitting with Constraints')
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print("Optimization was unsuccessful.")
    print("Message: ", result_with_bounds.message)

result_with_bounds.success, result_with_bounds.status, result_with_bounds.message
