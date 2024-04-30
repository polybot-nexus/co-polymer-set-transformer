# helper functions to get the molecular fingerprint representation format
import os
import pandas as pd
import numpy as np
import os
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestRegressor
import joblib
from scipy.spatial import distance
import ast
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from itertools import combinations
import colormath
from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie2000
from scipy.stats import norm
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


def abs_to_lab(data):
    """
    Description: Reading a csv file with the Absorption spectra and converting to the Lab values

    Parameters:
        file_name: the complete path and name of the csv file with the absorption spectra
        the dataframe includes a column named 'wavelength' and columns with the abs spectra named after the
        positions on the plate reader, e.g. 'C3', 'C4' etc.

    Returns:
        df: a pandas dataframe with the Lab color coordicates    """

    import pandas as pd
    import csv
    import os
    import json
    import numpy as np
    import colour # requires to install the Corol library: pip install colour

    # file_name = data.get('csv_name')
    # scaler = MinMaxScaler()

    lab_list = []
    # file_name = pd.read_csv(filename)
    # file_name = file_name.dropna(axis=0)
    # print('file_name', file_name)
    # scaled_data = scaler.fit_transform(file_name.iloc[: , 1:])
    scaled_df = pd.DataFrame(data).T
    file_name = pd.concat([pd.DataFrame(ast.literal_eval(electrochromics.wavelength.values[0]), columns=['wavelength']), scaled_df], axis=1)

    for col in file_name.columns.values[1:]:
        Trans= 10**(2-file_name[col].values)
        data_sample = dict(zip(file_name['wavelength'], Trans/100))
        sd = colour.SpectralDistribution(data_sample)
        cmfs = colour.MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
        illuminant = colour.SDS_ILLUMINANTS['D65']
        XYZ = colour.sd_to_XYZ(sd,cmfs,  illuminant)
        Lab = colour.XYZ_to_Lab(XYZ / 100)
        lab_list.append(Lab)

    return lab_list 

def tecan_read(filename):
    """
    Extracts raw data from Mangelan asc file into a csv file

    :param str filename: filename of Mangelan asc file to convert

    output: path of new csv file (str)

    """
    # import os
    # import csv
    # delimiter='\t'
    # data = {'wavelength': []}
    # column_names = set()

    # with open(filename, 'r') as input_file:
    #     for line in input_file:
    #         #print(line)
    #         line = line.strip()

    #         if line.startswith('*'):
    #             #print(line)
    #             fields = line[2:].replace('nm', '').split(delimiter)
    #             data['wavelength'].extend(fields)

    #         elif line.startswith('A'):
    #             print(line)
    #             columns = line.split(delimiter)
    #             column_name = columns[0]
    #             column_values = columns[1:]
    #             data[column_name] = column_values
    #             column_names.add(column_name)
            
    #         elif line.startswith('B'):
    #             columns = line.split(delimiter)
    #             column_name = columns[0]
    #             column_values = columns[1:]
    #             data[column_name] = column_values
    #             column_names.add(column_name)

    #         elif line.startswith('C'):
    #             columns = line.split(delimiter)
    #             column_name = columns[0]
    #             column_values = columns[1:]
    #             data[column_name] = column_values
    #             column_names.add(column_name)

    #         elif line.startswith('D'):
    #             columns = line.split(delimiter)
    #             column_name = columns[0]
    #             column_values = columns[1:]
    #             data[column_name] = column_values
    #             column_names.add(column_name)

    #         elif line.startswith('E'):
    #             columns = line.split(delimiter)
    #             column_name = columns[0]
    #             column_values = columns[1:]
    #             data[column_name] = column_values
    #             column_names.add(column_name)

    #         elif line.startswith('F'):
    #             columns = line.split(delimiter)
    #             column_name = columns[0]
    #             column_values = columns[1:]
    #             data[column_name] = column_values
    #             column_names.add(column_name)

    # if os.path.exists(filename):
    #     asc_basename = os.path.splitext(os.path.basename(filename))[0]
    #     csv_filename = asc_basename + ".csv"
    #     csv_filepath = filename.replace(os.path.basename(filename), csv_filename)

    # with open(csv_filepath, 'w', newline='') as output_file:
    #     csv_writer = csv.writer(output_file)
    #     csv_writer.writerow(['wavelength'] + sorted(column_names))
    #     num_rows = max(len(data['wavelength']), max(len(values) for values in data.values() if isinstance(values, list)))
    #     for i in range(num_rows):
    #         row = [data[column][i] if i < len(data[column]) else '' for column in ['wavelength'] + sorted(column_names)]
    #         #print(row)
    #         csv_writer.writerow(row)
            
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    scaler = MinMaxScaler()
    
    file_name = pd.read_csv('polybot_app/demo_files/from_tecan/green_loop_3.csv')  #(csv_filepath)
    print('file_name', file_name)
    scaled_data = scaler.fit_transform(file_name.iloc[: , 1:])
    scaled_df = pd.DataFrame(scaled_data, columns=file_name.columns[1:])
    file_name = pd.concat([file_name['wavelength'], scaled_df], axis=1)
    file_name.to_csv('polybot_app/demo_files/from_tecan/green_loop_3_scaled.csv', index=None)
    #return csv_filepath


def tecan_proc(filename):
    """
    Description: Reading a csv file with the Absorption spectra and converting to the Lab values

    Parameters:
        file_name: the complete path and name of the csv file with the absorption spectra
        the dataframe includes a column named 'wavelength' and columns with the abs spectra named after the 
        positions on the plate reader, e.g. 'C3', 'C4' etc.

    Returns:
        df: a pandas dataframe with the Lab color coordicates    """

    import pandas as pd
    import csv
    import os
    import json
    import numpy as np 
    import colour # requires to install the Corol library: pip install colour

    # file_name = data.get('csv_name')
    scaler = MinMaxScaler()
    
    lab_list = []
    file_name = pd.read_csv(filename)
    file_name = file_name.dropna(axis=0)
    print('file_name', file_name)
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

def smile_to_bits(smile):
  mol = Chem.MolFromSmiles(smile)
  return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024, useChirality=True)  

def get_vectors(smiles):
  bits = []
  for smile in smiles:
    try:
      bits.append(np.asarray(smile_to_bits(smile)))
    except:
      bits.append(np.zeros(1024))
  return bits

def bits_to_df(smiles, prefix):
  df = pd.DataFrame(get_vectors(smiles))
  columns = [f'{prefix}_{i}' for i in df.columns]
  df.columns = columns
  return df

def get_dft_descriptors_dictionary(dft_calculations_file):
    # create a dictionary to assign the molecular features to each of the smiles stings
    data_dft = pd.read_csv(dft_calculations_file)
    data_dft = data_dft.drop(['stoichiometry','number_of_atoms','charge','multiplicity', 'E_scf', 'zero_point_correction', 'E_thermal_correction',
        'H_thermal_correction', 'G_thermal_correction', 'E_zpe', 'E', 'H','converged',
        'G', ], axis=1) 
    scaler = MinMaxScaler()
    cols_to_scale = data_dft.columns[1:]
    # scale the selected columns
    data_dft[cols_to_scale] = scaler.fit_transform(data_dft[cols_to_scale])
    dictionary = data_dft.set_index('smiles').agg(list, axis=1).to_dict()
    return dictionary, data_dft.columns.values[1:]

def smile_to_dft(smile):
  dictionary , _= get_dft_descriptors_dictionary('datasets/dft_descriptors_ECPs.csv')
  return dictionary[smile]

def dft_descr(smiles):
  dictionary , descriptor_names= get_dft_descriptors_dictionary('datasets/dft_descriptors_ECPs.csv')
  bits = []
  for smile in smiles:
    try:
      bits.append(np.asarray(smile_to_dft(smile)))
    except:
      bits.append(np.zeros(len(descriptor_names)))
  return bits

def dft_descr_from_df(smiles, prefix):
  dictionary , descriptor_names= get_dft_descriptors_dictionary('datasets/dft_descriptors_ECPs.csv')
  df = pd.DataFrame(dft_descr(smiles))

  df.columns =[f'{prefix}_{i}' for i in descriptor_names] # descriptor_names
  return df

def create_predictions_dataset_dft(inventory, num_monomers, ratio_step):
    """Given the inventory as a list, create a dataset of all possible combinations
       and ratios of the first molecule of the list with the remaining ones.
       This dataset is the input to the trained predictive model.

    Parameters:
        inventory: SMILES list of molecules available in Chemspeed
        num_monomers: the number of monomers to be combined, can be 2,3, or both
        ratio_step: the step of the ratio of the first monomer, which will define the
                    ratios of the subsequent monomers. The sum of all the monomer ratios should be 100
    Returns:
        df: a pandas dataframe with all the possible pairs of the inventory
    """
    first_monomer = inventory[0]
    rest_monomers = inventory[1:]

    dataset=[]
    if 2 in num_monomers:       
       two_monomer_combinations = [[first_monomer, mol, 0] for mol in rest_monomers]
       for ratio in range(100, 49, -ratio_step):
          for combination in two_monomer_combinations:
             row ={
                "Combination": combination,
                "Ratio": [ratio/100, (100-ratio)/100, 0]
             }
             dataset.append(row)

    if 3 in num_monomers:     
       three_monomer_combinations = [[first_monomer] + list(mol_pair) for mol_pair in combinations(rest_monomers, 2)]
       for ratio in range(100, 49, -ratio_step):
          for combination in three_monomer_combinations:
             remaining_ratio = 100-ratio
             for ratio2 in range(remaining_ratio, -1, -ratio_step):
                ratio3 = remaining_ratio - ratio2
                row ={
                    "Combination": combination,
                    "Ratio": [ratio/100, ratio2/100, ratio3/100]
                }
                dataset.append(row)

    dataset = pd.DataFrame(dataset)
    dataset[["smiles1", "smiles2", "smiles3"]] = dataset["Combination"].apply(pd.Series)
    dataset[["percentage_1", "percentage_2", "percentage_3"]] = dataset["Ratio"].apply(pd.Series)
    dataset = dataset.drop(["Combination", "Ratio"], axis=1)
    dataset=dataset.fillna(0)
    dataset = dataset[dataset["percentage_1"] !=1]
    dataset=dataset.reset_index(drop=True)
    validation_1 = bits_to_df(dataset['smiles1'], 'bit_1')
    validation_2 = bits_to_df(dataset['smiles2'], 'bit_2')
    validation_3 = bits_to_df(dataset['smiles3'], 'bit_3')
    df1_dft = dft_descr_from_df(dataset['smiles1'], 'A')
    df2_dft  = dft_descr_from_df(dataset['smiles2'], 'B')
    df3_dft  = dft_descr_from_df(dataset['smiles3'], 'C')

    validation_dataset = pd.concat([
        dataset['smiles1'],
        dataset['smiles2'],
        dataset['smiles3'],
        validation_1,df1_dft,
        dataset[['percentage_1']],
        validation_2,df2_dft,
        dataset[['percentage_2']],
        validation_3, df3_dft,
        dataset[['percentage_3']]
    ], axis=1)            
    return validation_dataset


def get_train_data_representation(dataframe):   
   """Given the dataframe with the experimental results, i.e., smiles + ratios + extracted Lab values
      return the finger print representation"""
   smiles_A = 'Cc1sc(C)c2OCC(COCC(CC)CCCC)(COCC(CC)CCCC)COc12'
   df1 = bits_to_df(np.repeat(smiles_A, dataframe.shape[0],axis=0), 'bit_1')
   df2 = bits_to_df(dataframe.smiles2, 'bit_2')
   df3 = bits_to_df(dataframe.smiles3, 'bit_2')#(np.repeat(0, dataframe.shape[0],axis=0), 'bit_3')
   percentage_2 = dataframe.percentage_2 *100
   percentage_1 = 100 - dataframe.percentage_2
   percentage_3 = np.zeros(dataframe.shape[0])
   dataset = pd.concat([df1,pd.DataFrame(percentage_1.values, columns=['percentage_1']),df2,
                        pd.DataFrame(percentage_2.values, columns=['percentage_2']),
                        df3, pd.DataFrame(percentage_3, columns=['percentage_3'])], axis=1)
   return dataset

def get_train_data_representation_dft(dataframe):   
   """Given the dataframe with the experimental results, i.e., smiles + ratios + extracted Lab values
      return the finger print representation"""
   smiles_A = 'Cc1sc(C)c2OCC(COCC(CC)CCCC)(COCC(CC)CCCC)COc12'
   df1 = bits_to_df(np.repeat(smiles_A, dataframe.shape[0],axis=0), 'bit_1')
   df2 = bits_to_df(dataframe.smiles2, 'bit_2')
   df3 = bits_to_df(dataframe.smiles3, 'bit_3')#(np.repeat(0, dataframe.shape[0],axis=0), 'bit_3')
   df1_dft = dft_descr_from_df(np.repeat(smiles_A, dataframe.shape[0],axis=0), 'bit_1')
   df2_dft  = dft_descr_from_df(dataframe.smiles2, 'bit_2')
   df3_dft  = dft_descr_from_df(dataframe.smiles3, 'bit_3')   
   percentage_1 = dataframe.percentage_1
   percentage_2 = dataframe.percentage_2 
   percentage_3 = dataframe.percentage_3
   dataset = pd.concat([df1,df1_dft ,pd.DataFrame(percentage_1.values, columns=['percentage_1']),df2,df2_dft ,
                        pd.DataFrame(percentage_2.values, columns=['percentage_2']),
                        df3,df3_dft , pd.DataFrame(percentage_3, columns=['percentage_3'])], axis=1)
   return dataset

def select_next_exp_ml(dataframe, selected_color, uncertainties, iteration): 
    """Color comparison based on the delta-E value: https://python-colormath.readthedocs.io/en/latest/delta_e.html"""
    selected_color_lab = LabColor(selected_color[0], selected_color[1], selected_color[2])
    print('selected_color_lab', selected_color_lab)
    #print('dataframe',dataframe)
    dataframe['delta_e'] = dataframe.apply(lambda row: delta_e_cie2000(selected_color_lab, LabColor(row['L'], row['a'] ,row['b'])), axis=1)
    
    if iteration == 0 or iteration ==1:
       EI_values = expectedImprovement(dataframe['delta_e'].values, uncertainties, ybest=0, epsilon=1)
       #upperConfidenceBound(dataframe['delta_e'].values, uncertainties, epsilon=0.01)
    else: 
       EI_values = expectedImprovement(dataframe['delta_e'].values, uncertainties, ybest=0, epsilon=0.1)
    
    #print('uncertainties', uncertainties)
    #print('index', dataframe.index)
    
    # top_6_loc = np.argpartition(EI_values, 6)[:6] # we want to select the top 6 experiments with expected improvent
    #
    top_6_loc = np.argpartition(EI_values, 6)[:6]
    print('top_6_loc', top_6_loc )
    top_6_sorted_indices = [np.argsort(EI_values[top_6_loc])]
    print('top_6_sorted_indices', top_6_sorted_indices)
    df = dataframe.loc[top_6_loc , :]
    dataframe['uncertainties'] = uncertainties
    dataframe['EI_values'] = EI_values
    print('top_6_loc', top_6_loc)
    plt.scatter(dataframe.index, uncertainties)
    plt.scatter(top_6_loc,dataframe.loc[top_6_loc, :]['uncertainties'],color='r',marker='s',s=100,facecolor='None')
    plt.xlabel('index', fontsize=14)
    plt.ylabel('Uncertainty', fontsize=14)
    plt.show()
    print('EI_values', EI_values.min() )
    plt.scatter(dataframe.index, EI_values)
    plt.scatter(top_6_loc,dataframe.loc[top_6_loc, :]['EI_values'],color='r',marker='s',s=100,facecolor='None')
    plt.xlabel('index', fontsize=14)
    plt.ylabel('Expected improvement', fontsize=14)
    plt.show()

    plt.scatter(EI_values, uncertainties)
    plt.scatter(dataframe.loc[top_6_loc, :]['EI_values'],dataframe.loc[top_6_loc, :]['uncertainties'],color='r',marker='s',s=100,facecolor='None')
    plt.xlabel('Expected improvement', fontsize=14)
    plt.ylabel('Uncertainty', fontsize=14)
    plt.show()
    # print('new_df', dataframe.loc[top_6_loc, :])
    # dataframe.sort_values(by=['delta_e'], ascending =True).head(6) # select the 6 points closer to the selected color value
    #df.to_csv('test.csv')
    return df

# Acquisition functions
def upperConfidenceBound(ye_pred, esigma, epsilon):
    """
        ye_pred: predicted values distance to target
        esigma: uncertainty
        epsilon: control exploration/exploitation. Higher epsilon means more exploration
    """
    ucb = np.empty(ye_pred.size, dtype=float)
    for ii in range(0,ye_pred.size):
        if esigma[ii] > 0:
            ucb[ii]=(ye_pred[ii]+epsilon*(esigma[ii]))
        else:
            ucb[ii]=0.0
    return ucb

def expectedImprovement(ye_pred, esigma, ybest, epsilon=0.01):  
    expI = np.empty(ye_pred.size, dtype=float)     
    for ii in range(ye_pred.size):         
        if esigma[ii] > 0:             
            zzval=(ye_pred[ii]-ybest)/float(esigma[ii])             
            expI[ii]=(ye_pred[ii]-ybest-epsilon)*norm.cdf(zzval)+esigma[ii]*norm.pdf(zzval)         
        else:             
            expI[ii]=0.0     
    return expI

def convert_for_chemspeed(dataframe):
    monomers_dict = {
       "Cc1sc(C)c2OCC(COCC(CC)CCCC)(COCC(CC)CCCC)COc12":"ProDOT-2Br", 
       "C1=CC(=CC=C1C)C":"Ben-2Br", 
       "Cc1ccc(C)c2nsnc12":"BTD-2Br", 
       "Cc1sc(C)c(OC)c1OC":"DMoT-2Br", #Cc1sc(C)c(OC)c1OC	
       "Cc1sc(C)c2OCCOc12":"EDOT-2Br",
       "Cc1ccc3c(c1)Cc2cc(C)ccc23":"green-2Br",
    }
    dataframe['percentage_1'] = (dataframe['percentage_1']-0.5)*2
    dataframe['percentage_2'] = dataframe['percentage_2']*2
    dataframe['percentage_3'] = dataframe['percentage_3']*2
    csv = pd.DataFrame(0, index=np.arange(dataframe.shape[0]), columns=["ProDOT-2Br", "Ben-2Br", "BTD-2Br", "DMoT-2Br", "EDOT-2Br"])
    for i, ix in enumerate(dataframe.index):
       for monomer in ['1','2','3']:
            label = monomers_dict.get(dataframe.loc[ix, f'smiles{monomer}'], 0)
            value = dataframe.loc[ix, f'percentage_{monomer}'] * 1000
            if label in csv.columns:
                csv.loc[i, label] = round(value)
    return csv

def peak_check(df):
   #### We check one by one the abs spectra and create a csv with the  0,1,-1 indicating if the samples need :
    ## 1: too concentrated abs_peak>2.4
    ## 0: normal
    ## -1: too dilute abs_max<0.2
   
   abs_spectra = pd.read_csv(df)
   wavelength = abs_spectra['wavelength']
   peak_cluster = []
   for col in abs_spectra.columns[1:]:
      
      absorbance = abs_spectra[col].values
      peaks,properties= find_peaks(absorbance, height=0.1)
      print('peaks', len(peaks))

      #Identify the highest peak     
      if len(peaks) > 0:
        highest_peak_idx = np.argmax(properties['peak_heights'])
        highest_peak_wavelength = wavelength[peaks[highest_peak_idx]]
        highest_peak_absorbance = properties['peak_heights'][highest_peak_idx]
        print('highest_peak_absorbance', highest_peak_absorbance)

        if highest_peak_absorbance>= 2.5:
                peak_cluster.append(1)
        elif highest_peak_absorbance<= 0.3:
                peak_cluster.append(-1)
        else:
                peak_cluster.append(0)
       
      else:
        peak_cluster.append(-1)
           
   print(peak_cluster)
   return peak_cluster
                
def get_lab_distance(target_Lab, measured_lab_values):
    """Color comparison based on the delta-E value: https://python-colormath.readthedocs.io/en/latest/delta_e.html"""
    target_Lab_color = LabColor(*target_Lab)
    distances = np.array([delta_e_cie2000(target_Lab_color, LabColor(*val)) for val in measured_lab_values])
    print('distances', distances )
    if np.any(distances < 5):
       return True