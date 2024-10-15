#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from colormath.color_diff import delta_e_cie2000
from colormath.color_objects import LabColor
import os
import csv

def tecan_read(filename):
    """
    Extracts raw data from Mangelan asc file into a csv file

    :param str filename: filename of Mangelan asc file to convert

    output: path of new csv file (str)

    """
    import os
    import csv
    delimiter='\t'
    data = {'wavelength': []}
    column_names = set()

    with open(filename, 'r') as input_file:
        for line in input_file:
            line = line.strip()

            if line.startswith('*'):
                fields = line[2:].replace('nm', '').split(delimiter)
                data['wavelength'].extend(fields)

            elif line.startswith('A'):
                print(line)
                columns = line.split(delimiter)
                column_name = columns[0]
                column_values = columns[1:]
                data[column_name] = column_values
                column_names.add(column_name)
            
    if os.path.exists(filename):
        asc_basename = os.path.splitext(os.path.basename(filename))[0]
        csv_filename = asc_basename + ".csv"
        csv_filepath = filename.replace(os.path.basename(filename), csv_filename)

    with open(csv_filepath, 'w', newline='') as output_file:
        csv_writer = csv.writer(output_file)
        csv_writer.writerow(['wavelength'] + sorted(column_names))
        num_rows = max(len(data['wavelength']), max(len(values) for values in data.values() if isinstance(values, list)))
        for i in range(num_rows):
            row = [data[column][i] if i < len(data[column]) else '' for column in ['wavelength'] + sorted(column_names)]
            csv_writer.writerow(row)
            
    
    scaler = MinMaxScaler()
    
    file_name = pd.read_csv(csv_filepath) 
    scaled_data = scaler.fit_transform(file_name.iloc[: , 1:])
    scaled_df = pd.DataFrame(scaled_data, columns=file_name.columns[1:])
    file_name = pd.concat([file_name['wavelength'], scaled_df], axis=1)
    return file_name


def tecan_proc(filename):
    """
    Description: Reading a csv file with the Absorption spectra and converting to the Lab values

    Parameters:
        file_name: the complete path and name of the csv file with the absorption spectra
        the dataframe includes a column named 'wavelength' and columns with the abs spectra named after the 
        positions on the plate reader, e.g. 'C3', 'C4' etc.

    Returns:
        df: a pandas dataframe with the Lab color coordicates    """


    import colour # requires to install the Corol library: pip install colour

    # file_name = data.get('csv_name')
    scaler = MinMaxScaler()
    
    lab_list = []
    file_name = pd.read_csv(filename)
    file_name = file_name.dropna(axis=0)
    scaled_data = scaler.fit_transform(file_name.iloc[: , 1:])
    scaled_df = pd.DataFrame(scaled_data, columns=file_name.columns[1:])
    file_name = pd.concat([file_name['wavelength'], scaled_df], axis=1)

    for col in file_name.columns.values[1:]:   
        Trans= 10**(2-file_name[col].values)
        data_sample = dict(zip(file_name['wavelength'], Trans/100))
        sd = colour.SpectralDistribution(data_sample)
        cmfs = colour.MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
        illuminant = colour.SDS_ILLUMINANTS['D65']
        XYZ = colour.sd_to_XYZ(sd,cmfs,  illuminant)
        Lab = colour.XYZ_to_Lab(XYZ / 100)
        lab_list.append(Lab)
  
    #df = pd.DataFrame(file_name['wavelength'], pd.DataFrame(lab_list, columns = file_name.columns.values[1:]))
    
    return lab_list #df 


def get_lab_distance(target_Lab, measured_lab_values):
    """Color comparison based on the delta-E value: https://python-colormath.readthedocs.io/en/latest/delta_e.html"""
    target_Lab_color = LabColor(*target_Lab)
    distances = np.array([delta_e_cie2000(target_Lab_color, LabColor(*val)) for val in measured_lab_values])
    print('distances', distances)
    if np.any(distances < 5):
       return True
    
local_path_from_tecan = "C:/Users/kvriz/Desktop/Polybot_ECPs/utils/"
fname_tecan = f"Loop-3-20240402.asc"  
filename=os.path.join(local_path_from_tecan, fname_tecan)
df = tecan_read(filename)
lab_values = tecan_proc(df)
# pd.DataFrame(lab_values, columns=["L", "a", "b"]).to_csv('polybot_app/demo_files/metadata/green_loop_3.csv')
print('lab_values',lab_values)



# Example of measuring the deltaE of two Lab values
target_Lab= np.array([60, -15, 40])
measured_lab_values= [[60,  -9.7625456 ,  20.17920844]]
print(get_lab_distance(target_Lab, measured_lab_values))

