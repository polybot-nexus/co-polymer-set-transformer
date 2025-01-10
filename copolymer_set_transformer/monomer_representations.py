from rdkit.Chem import rdFingerprintGenerator
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def smile_to_bits(smile):
  mol = Chem.MolFromSmiles(smile)
  if mol == '0':
    return np.zeros(1024)
  fpgen1 = rdFingerprintGenerator.GetMorganGenerator(radius=3,fpSize=1024,countSimulation=True)
  return fpgen1.GetCountFingerprintAsNumPy(mol)

def get_vectors(smiles):
  bits = []
  for smile in smiles:
    if smile ==0 or smile == '0':
      bits.append(np.zeros(1024))
    else:
      bits.append(np.asarray(smile_to_bits(smile)))
  return bits

def bits_to_df(smiles, prefix):
  df = pd.DataFrame(get_vectors(smiles))
  columns = [f'{prefix}_{i}' for i in df.columns]
  df.columns = columns
  return df

def get_dft_descriptors_dictionary(dft_calculations_file):
    """Create the DFT descriptors dictionary where each SMILES string is associated with the DFT values calculated 
    using Auto-QChem workflow"""
    data_dft = pd.read_csv(dft_calculations_file)
    data_dft = data_dft.drop(['stoichiometry','number_of_atoms','charge','multiplicity', 'E_scf', 'zero_point_correction', 'E_thermal_correction',
        'H_thermal_correction', 'G_thermal_correction', 'E_zpe', 'E', 'H','converged',
        'G', ], axis=1)
    scaler = MinMaxScaler()

    cols_to_scale = data_dft.columns[1:]

    data_dft[cols_to_scale] = scaler.fit_transform(data_dft[cols_to_scale])

    dictionary = data_dft.set_index('smiles').agg(list, axis=1).to_dict()
    return dictionary, data_dft.columns.values[1:]


def dft_descr_from_df(smiles, prefix):
  df = pd.DataFrame(dft_descr(smiles))

  df.columns =[f'{prefix}_{i}' for i in descriptor_names] 
  return df

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


def smile_to_bits(smile):
  mol = Chem.MolFromSmiles(smile)
  fpgen1 = rdFingerprintGenerator.GetMorganGenerator(radius=3,fpSize=1024,countSimulation=True)
  return fpgen1.GetCountFingerprintAsNumPy(mol)


def get_train_data_representation_dft(dataframe):
  """Function to create the full vector representation of a dataset with electrochromic polymers.
  The dataframe consists of the rdkit molecular fingerprint and the DFT desctiptors."""

  df1 = bits_to_df(dataframe.smiles1, 'bit_1')
  df2 = bits_to_df(dataframe.smiles2, 'bit_2')
  df3 = bits_to_df(dataframe.smiles3, 'bit_3')
  df1_dft = dft_descr_from_df(dataframe.smiles1, 'A')
  df2_dft  = dft_descr_from_df(dataframe.smiles2, 'B')
  df3_dft  = dft_descr_from_df(dataframe.smiles3, 'C')

  dataset = pd.concat([df1,df1_dft ,pd.DataFrame(dataframe[['percentage_1']].values, columns=['percentage_1']),df2,df2_dft ,
                        pd.DataFrame(dataframe[['percentage_2']].values, columns=['percentage_2']),
                        df3,df3_dft , pd.DataFrame(dataframe[['percentage_3']].values, columns=['percentage_3'])], axis=1)                    
  return dataset


dictionary , descriptor_names= get_dft_descriptors_dictionary('datasets/dft_descriptors_ECPs.csv')