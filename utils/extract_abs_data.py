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
    peaks= find_peaks_cwt(intensities, [5])
    return peaks# properties['peak_heights']#.max()


def export_data(folder_name, file_name):
    filename = os.path.join(folder_name, file_name)
    data = pd.read_csv(filename, delimiter='\t')#, asty)
    name = data.iloc[0,1]
    x,y = list(data.iloc[1:,0].values)[::-1], [float(i) for i in data.iloc[1:,1].values][::-1]
    peak_locs, peak_HWHMs, intensities = find_peaks_HWHM(x,y)
    indexes_1 = np.where(peak_locs > 420)
    indexes_2 = np.where(intensities > 0.15)

    is_in_array2 = np.isin(indexes_1, indexes_2)
    common_elements = set(np.array(indexes_1)[is_in_array2])
    ind = np.array(peak_locs)[list(common_elements)]
    ind_1 = np.array(intensities)[list(common_elements)]
    ind_2 = np.array(peak_HWHMs)[list(common_elements)]

    return ind, ind_1, ind_2, name


# specify the parent directory path
parent_directory_path = "."

oleic_oleyl=[]
oleic_pb=[]
Cs_Pb=[]
peak_wavelengths=[]
peak_intensity=[]
hwhm=[]


for root, directories, files in os.walk(parent_directory_path):

    # iterate over all the files in the directory
    for filename in files:

        if filename.endswith(".txt"):

            file_path = os.path.join(root, filename)
            ind, ind_1, ind_2, name = export_data(root, filename)
            oleic_oleyl.append(root)
            oleic_pb.append(filename.split('.')[0])
            Cs_Pb.append(name)
            peak_wavelengths.append(ind)
            peak_intensity.append(ind_1)
            hwhm.append(ind_2)
            #print(root, filename, name, ind, ind_1, ind_2)
 
dataset = pd.concat([pd.DataFrame(oleic_oleyl, columns=['oleic_oleyl']), pd.DataFrame(oleic_pb, columns=['oleic_pb']),
           pd.DataFrame(Cs_Pb, columns=['Cs_Pb']), pd.DataFrame(peak_wavelengths),
            pd.DataFrame(peak_intensity), pd.DataFrame(hwhm) ], axis=1)

dataset.to_csv('extracted_dataset.csv', index=None)