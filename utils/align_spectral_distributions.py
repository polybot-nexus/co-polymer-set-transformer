from colour.plotting import *
import colour
import matplotlib.pyplot as plt
import pandas as pd
import csv
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import ast 
# file_path = "C:/Users/kvriz/Desktop/Polybot_ECPs/datasets/absorption_spectra/literature_data/Adv. Mater.2010,22,724–728/P4b.csv"

def align_abs_spectra(file_path):
    df = pd.read_csv(file_path)
    df = df.sort_values(by='Wavelength (nm)')

    # df = pd.read_csv(file_path, delimiter='\t')
    # df = pd.read_csv(file_path,sep=' ')#, delimiter='\t')#, dtype=float)
    # print(df)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    # print(df.columns.values) 
    # data_dict = df.set_index("Wavelength")["(nm)\tAbsorbance"].to_dict() 
    data_dict = df.set_index('Wavelength (nm)')['Absorbance'].to_dict() 
    # print('data_dict', data_dict)
    sd = colour.SpectralDistribution(data_dict, name='PProDOT')

    # Copying the sample spectral distribution.
    sd_copy = sd.copy()
    low_wavelength = list(data_dict.keys())[0]
    high_wavelength = list(data_dict.keys())[-1]
    # print('wavelength',low_wavelength, high_wavelength)

    # Interpolating the copied sample spectral distribution.
    sd_copy.interpolate(colour.SpectralShape(low_wavelength, high_wavelength, 1))
    sd_copy.extrapolate(colour.SpectralShape(350, 800, 1))
    
    sd_copy.interpolate(colour.SpectralShape(350, 800, 1))
    #print('len sd', len(sd_copy))
    # plot_single_sd(sd)
    # plot_multi_sds([sd, sd_copy])
    return sd_copy


# sd_copy = align_abs_spectra(file_path)
# data_frame = pd.DataFrame(sd_copy.to_series()).reset_index()        
# data_frame.columns = ['Wavelength (nm)', 'Absorbance']
# scaler = MinMaxScaler()
# scaled_data = scaler.fit_transform(data_frame.iloc[: , 1:])
# scaled_df = pd.DataFrame(scaled_data, columns=['Absorbance'])
# file_name = pd.concat([data_frame['Wavelength (nm)'], scaled_df], axis=1)
# plt.plot(file_name["Wavelength (nm)"], file_name["Absorbance"])
# plt.show()

# file_name.to_csv(f"{root}/{filename.rsplit('.', 1)[0]}_aligned.csv", index=None)
# print(align_abs_spectra('C:/Users/kvriz/Desktop/Polybot_ECPs/datasets/absorption_spectra/in_house/batch_3_spectra/50-A4-10-A6-40-B3.txt'))
parent_directory_path = 'C:/Users/kvriz/Desktop/Polybot_ECPs/datasets/absorption_spectra/literature_data'

def create_aligned_files_literature(parent_directory_path):
    for root, directories, files in os.walk(parent_directory_path):
        for filename in files:
            #try:
                if filename.endswith(".csv"):
                    file_path = os.path.join(root, filename)
                    scaler = MinMaxScaler()
                    print('file_path', file_path)            
                    sd_copy = align_abs_spectra(file_path)
                    data_frame = pd.DataFrame(sd_copy.to_series()).reset_index()        
                    data_frame.columns = ['Wavelength (nm)', 'Absorbance']
                    scaled_data = scaler.fit_transform(data_frame.iloc[: , 1:])
                    scaled_df = pd.DataFrame(scaled_data, columns=['Absorbance'])
                    file_name = pd.concat([data_frame['Wavelength (nm)'], scaled_df], axis=1)
                    # plt.plot(file_name["Wavelength (nm)"], file_name["Absorbance"])
                    # plt.show()
                    file_name.to_csv(f"{root}/{filename.rsplit('.', 1)[0]}_aligned.csv", index=None)
            # except:
            #     print(root , filename)


def clean_aligned_files(parent_directory_path):
    for root, directories, files in os.walk(parent_directory_path):
        for filename in files:
            if filename.endswith("aligned.csv"):
                file_path = os.path.join(root, filename)
                os.remove(file_path)
        

# parent_directory_path = 'C:/Users/kvriz/Desktop/Polybot_ECPs/datasets/absorption_spectra/in_house'

# for root, directories, files in os.walk(parent_directory_path):
#    for filename in files:
#        if filename.endswith(".txt"):
#         try:
#             file_path = os.path.join(root, filename)
#             scaler = MinMaxScaler()            
#             sd_copy = align_abs_spectra(file_path)
#             data_frame = pd.DataFrame(sd_copy.to_series()).reset_index()        
#             data_frame.columns = ['Wavelength (nm)', 'Absorbance']
#             scaled_data = scaler.fit_transform(data_frame.iloc[: , 1:])
#             scaled_df = pd.DataFrame(scaled_data, columns=['Absorbance'])
#             file_name = pd.concat([data_frame['Wavelength (nm)'], scaled_df], axis=1)
#             file_name.to_csv(f"{root}/{filename.rsplit('.', 1)[0]}_aligned.csv", index=None)
#         except: 
#            print(filename)


# parent_directory_path = 'C:/Users/kvriz/Desktop/Polybot_ECPs/datasets/absorption_spectra/'
def get_all_abs_from_files(parent_directory_path):
    polymer_name = []
    wavelength_vector = []
    intensity_vector = []
    source = []

    for root, directories, files in os.walk(parent_directory_path):
        for filename in files:
            if filename.endswith("_aligned.csv"):
                source.append(os.path.basename(root))
                polymer_name.append(filename.rsplit('_', 1)[0])
                file_path = os.path.join(root, filename)
                df=pd.read_csv(file_path)
                
                # wavelength_vector.append([ast.literal_eval(df['Wavelength (nm)'].values[i]) for i in range(df['Wavelength (nm)'].shape[0])])
                # intensity_vector.append([ast.literal_eval(df['Absorbance'].values[i]) for i in range(df['Absorbance'].shape[0])])
                
                # print(", ".join(map(str, df['Wavelength (nm)'].values)))

                wavelength_vector.append([", ".join(map(str, df['Wavelength (nm)'].values))])
                intensity_vector.append([", ".join(map(str, df['Absorbance'].values))])

    # xaxis = df['Wavelength (nm)'].values
    # xaxis = ast.literal_eval(xaxis[0])
    # intensity = df['Absorbance'].values
    # intensity =ast.literal_eval(intensity[0])
    # for i in range(len(xaxis[:])):
    #     cleaned_string = xaxis[i].strip('[ ]').replace('\n', ' ')
    #     float_list = [float(num) for num in cleaned_string.split() if num not in ('', '.')]
    #     wavelength_vector.append(float_list)
    # # print(float_list)

    # for i in range(len(intensity[:])):
    #     cleaned_string = intensity[i].strip('[ ]').replace('\n', ' ')
    #     float_list = [float(num) for num in cleaned_string.split() if num not in ('', '.')]
    #     intensity_vector.append(float_list)

    dataset = pd.concat([pd.DataFrame(source, columns=['source']), pd.DataFrame(polymer_name, columns=['polymer_id']),
            pd.DataFrame(wavelength_vector, columns=['wavelength']), pd.DataFrame(intensity_vector, columns=['intensity'])], axis=1)

    dataset.to_csv('C:/Users/kvriz/Desktop/Polybot_ECPs/datasets/absorption_spectra/all_abs_spectra_corrected.csv', index=None)

# code to correct the format of saving the absorption spectra and intensity 
# abs_spectra =[]
# for i in range(len(xaxis[:])):
#   cleaned_string = xaxis[i].strip('[ ]').replace('\n', ' ')
#   float_list = [float(num) for num in cleaned_string.split() if num not in ('', '.')]
#   abs_spectra.append(float_list)
  # print(float_list)


##########Usage#############################################################################################################################

# Clean the folder from the aligned files
# clean_aligned_files(parent_directory_path)

# Create the aligned files from the literature dataset
# create_aligned_files_literature(parent_directory_path)

parent_directory_path = 'C:/Users/kvriz/Desktop/Polybot_ECPs/datasets/absorption_spectra/'
get_all_abs_from_files(parent_directory_path)