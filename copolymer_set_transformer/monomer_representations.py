from rdkit.Chem import rdFingerprintGenerator
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def smile_to_bits(smile):
  mol = Chem.MolFromSmiles(smile)
  fpgen1 = rdFingerprintGenerator.GetMorganGenerator(radius=3,fpSize=1024,countSimulation=True)
  return fpgen1.GetCountFingerprintAsNumPy(mol)

def get_vectors(smiles):
  paws = []
  for smile in smiles:
    try:
      paws.append(np.asarray(smile_to_bits(smile)))
    except:
      paws.append(np.zeros(1024))
  return paws

def bits_to_df(smiles, prefix):
  df = pd.DataFrame(get_vectors(smiles))
  columns = [f'{prefix}_{i}' for i in df.columns]
  df.columns = columns
  return df

def get_dft_descriptors_dictionary(dft_calculations_file):
    # create a dictionary to assign the molecular features to each of the smiles stings
    data_dft = pd.read_csv(dft_calculations_file)
    data_dft = data_dft.drop(['stoichiometry','number_of_atoms','charge','multiplicity', 'E_scf', 'zero_point_correction', 'E_thermal_correction',
        'H_thermal_correction', 'G_thermal_correction', 'E_zpe', 'E', 'H','converged',#'ES_<S**2>',
        'G', ], axis=1)
    scaler = MinMaxScaler()

    cols_to_scale = data_dft.columns[1:]

    data_dft[cols_to_scale] = scaler.fit_transform(data_dft[cols_to_scale])

    dictionary = data_dft.set_index('smiles').agg(list, axis=1).to_dict()
    return dictionary, data_dft.columns.values[1:]

def smile_to_dft(smile):
  return dictionary[smile]

def dft_descr(smiles):
  bits = []
  for smile in smiles:
    try:
      bits.append(np.asarray(smile_to_dft(smile)))
    except:
      bits.append(np.zeros(len(descriptor_names)))
  return bits

def dft_descr_from_df(smiles, prefix):
  df = pd.DataFrame(dft_descr(smiles))

  df.columns =[f'{prefix}_{i}' for i in descriptor_names] # descriptor_names
  return df

def smile_to_bits(smile):
  mol = Chem.MolFromSmiles(smile)
  fpgen1 = rdFingerprintGenerator.GetMorganGenerator(radius=3,fpSize=1024,countSimulation=True)
  return fpgen1.GetCountFingerprintAsNumPy(mol)

def get_train_data_representation_dft(dataframe):
   """Given the dataframe with the experimental results, i.e., smiles + ratios + extracted Lab values
      return the finger print representation"""
   df1 = bits_to_df(dataframe.smiles2, 'bit_1')
   df2 = bits_to_df(dataframe.smiles2, 'bit_2')
   df3 = bits_to_df(dataframe.smiles3, 'bit_3')
   df1_dft = dft_descr_from_df(dataframe.smiles1, 'bit_1')
   df2_dft  = dft_descr_from_df(dataframe.smiles2, 'bit_2')
   df3_dft  = dft_descr_from_df(dataframe.smiles3, 'bit_3')
   percentage_1 = dataframe.percentage_1
   percentage_2 = dataframe.percentage_2
   percentage_3 = dataframe.percentage_3
   dataset = pd.concat([df1,df1_dft ,pd.DataFrame(dataframe[['percentage_1']].values, columns=['percentage_1']),df2,df2_dft ,
                        pd.DataFrame(dataframe[['percentage_2']].values, columns=['percentage_2']),
                        df3,df3_dft , pd.DataFrame(dataframe[['percentage_3']].values, columns=['percentage_3'])], axis=1)                    
   return dataset


dictionary , descriptor_names= get_dft_descriptors_dictionary('/datasets/dft_descriptors_ECPs.csv')