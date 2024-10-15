from scipy.optimize import curve_fit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, find_peaks_cwt
import os

def find_peaks_HWHM(x, y):
    """Find all the peaks in the spectrum and calculate their half width half maximum.

    Parameters:
    x (array_like): The x values of the spectrum.
    y (array_like): The y values of the spectrum.

    Returns:
    A tuple of two arrays:
    - peak_locs: The x locations of the peaks.
    - peak_HWHMs: The half width half maximum of the peaks.
    """

    # Find all the peaks in the spectrum
    #peaks, _ = find_peaks(y)
    peaks = find_peaks_cwt(y, [5])
    # Calculate the half width half maximum of each peak
    peak_locs = np.array(x)[peaks]
    intensities = np.array(y)[peaks]
    peak_HWHMs = np.zeros_like(peak_locs)
    for i, peak_loc in enumerate(peak_locs):
        y_peak = y[peaks[i]]
        half_max = y_peak / 2
        left_idx = np.argmin(np.abs(np.array(y)[:peaks[i]] - half_max))
        right_idx = np.argmin(np.abs(np.array(y)[peaks[i]:] - half_max)) + peaks[i]
        peak_HWHMs[i] = x[right_idx] - x[left_idx]

    return peak_locs, peak_HWHMs, intensities

def abs_max_intensity(wavelength, intensities):
    """
    For a given Absorption spectra detect the maximum peak in a given range
    
    Args:
        dataset : The saved absorption spectra
        range: The range in nm, in which we expect the absorption peak for our material
    """
    #intensities =dataset.intensities
    peaks= find_peaks_cwt(intensities)#, [2])
    return peaks#.properties['peak_heights']#.max()


def find_peaks_HWHM(x, y, prominence=0.1):
    """Find major peaks in the spectrum and calculate their HWHM based on prominence.

    Parameters:
    x (array_like): The x values of the spectrum.
    y (array_like): The y values of the spectrum.
    prominence (float): The minimum prominence of peaks to consider.

    Returns:
    A tuple of three arrays:
    - peak_locs: The x locations of the peaks.
    - peak_HWHMs: The half width half maximum of the peaks.
    - intensities: The y values at the peaks.
    """

    # Find peaks with a prominence criteria
    peaks, properties = find_peaks(y, prominence=prominence)
    peak_locs = np.array(x)[peaks]
    intensities = np.array(y)[peaks]

    # Calculate HWHM for each peak
    peak_HWHMs = np.zeros_like(peak_locs)
    for i, peak in enumerate(peaks):
        half_max = intensities[i] / 2
        left_idx = np.argmin(np.abs(y[:peak] - half_max))
        right_idx = peak + np.argmin(np.abs(y[peak:] - half_max))
        peak_HWHMs[i] = x[right_idx] - x[left_idx]

    return peak_locs, peak_HWHMs, intensities


def export_data(folder_name, file_name):
    filename = os.path.join(folder_name, file_name)
    data = pd.read_csv(filename)
    name = data.iloc[0, 1]
    x, y = list(data.iloc[1:, 0].values)[::-1], [float(i) for i in data.iloc[1:, 1].values][::-1]

    peak_locs, peak_HWHMs, intensities = find_peaks_HWHM(x, y)

    # Plot the data
    plt.plot(x, y, label='Spectrum')

    # Mark the peaks
    plt.scatter(peak_locs, intensities, color='red', zorder=5, label='Peaks')

    # Annotate the peaks with their wavelength and visualize HWHM
    for i, peak_loc in enumerate(peak_locs):
        # Annotate peak wavelength
        plt.annotate(f'{peak_loc} nm', (peak_loc, intensities[i]), textcoords="offset points", xytext=(0,10), ha='center')

        # Visualize HWHM by drawing horizontal and vertical lines
        half_max = intensities[i] / 2
        hwhm_start = peak_loc - peak_HWHMs[i] / 2
        hwhm_end = peak_loc + peak_HWHMs[i] / 2

        # Draw HWHM lines
        plt.hlines(half_max, hwhm_start, hwhm_end, color='green', linestyles='dashed', label='HWHM' if i == 0 else "")
        plt.vlines([hwhm_start, hwhm_end], half_max - 0.05, half_max + 0.05, color='green', linestyles='dotted')

    plt.legend()
    plt.savefig('test_plot.svg')
    plt.show()

    return peak_locs, intensities, peak_HWHMs, name


# export_data("C:/Users/kvriz/Desktop/Polybot_ECPs\datasets/absorption_spectra/literature_data/ACS Macro Lett., 5, 714-717, 2016", "PProDOT_new.csv")



# Define a single Gaussian function
def gaussian(x, center, amplitude, width):
    return amplitude * np.exp(-((x - center) ** 2) / (2 * width ** 2))

# Function to model the sum of 20 Gaussian peaks
def sum_of_gaussians(x, *params):
    assert len(params) == 20 * 3  # Ensure we have 3 parameters for each of the 20 Gaussians
    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        center, amplitude, width = params[i:i+3]
        y += gaussian(x, center, amplitude, width)
    return y


folder_name = "C:/Users/kvriz/Desktop/Polybot_ECPs\datasets/absorption_spectra/literature_data/ACS Macro Lett., 5, 714-717, 2016"
file_name = "PProDOT_new.csv"
filename = os.path.join(folder_name, file_name)
data = pd.read_csv(filename)
name = data.iloc[0, 1]
x_data, y_data = list(data.iloc[1:, 0].values)[::-1], [float(i) for i in data.iloc[1:, 1].values][::-1]
# Example spectrum data (x and y values)
# x_data = np.linspace(350, 450, 400)  # wavelength range
# y_data = np.random.normal(size=x_data.size)  # simulated absorption values; replace with your actual spectrum
# plt.plot(x_data, y_data, label='Spectrum')
# plt.show()
# Initial guess for the parameters
# For simplicity, let's assume the peaks are evenly distributed and have the same initial amplitude and width
initial_guess = []
print(min(x_data), max(x_data))
num_peaks = 20
for i in range(num_peaks):
    center = np.linspace(min(x_data), max(x_data), num_peaks + 1)[i]
    amplitude = 1
    width = 1
    initial_guess += [center, amplitude, width]

# # Fit the model to the data
# # popt, pcov = curve_fit(sum_of_gaussians, x_data, y_data, p0=initial_guess)
bounds = (min(x_data), max(x_data))  # Example bounds: all parameters must be between 0 and infinity
popt, pcov = curve_fit(sum_of_gaussians, x_data, y_data, p0=initial_guess, maxfev=50000, bounds=bounds)

# Plot the fitted curve
plt.figure(figsize=(10, 6))
plt.plot(x_data, y_data, 'b-', label='Data')
plt.plot(x_data, sum_of_gaussians(x_data, *popt), 'r--', label='Fit')
plt.legend()
plt.show()

# The popt array contains the optimized parameters for all Gaussian peaks
# You can reshape this array to more easily interpret the parameters of each peak
# optimized_parameters = np.reshape(popt, (num_peaks, 3))

# print("Optimized parameters for each Gaussian peak (center, amplitude, width):")
# print(optimized_parameters)


#for root, directories, files in os.walk(parent_directory_path):
#    # iterate over all the files in the directory
#    for filename in files:
#        if filename.endswith(".txt"):
#            file_path = os.path.join(root, filename)
#            ind, ind_1, ind_2, name = export_data(root, filename)
#            oleic_oleyl.append(root)
#            oleic_pb.append(filename.split('.')[0])
#            Cs_Pb.append(name)
#            peak_wavelengths.append(ind)
#            peak_intensity.append(ind_1)
#            hwhm.append(ind_2)
            #print(root, filename, name, ind, ind_1, ind_2)
 
#dataset = pd.concat([pd.DataFrame(oleic_oleyl, columns=['oleic_oleyl']), pd.DataFrame(oleic_pb, columns=['oleic_pb']),
#           pd.DataFrame(Cs_Pb, columns=['Cs_Pb']), pd.DataFrame(peak_wavelengths),
#            pd.DataFrame(peak_intensity), pd.DataFrame(hwhm) ], axis=1)


#dataset.to_csv('extracted_dataset.csv', index=None)