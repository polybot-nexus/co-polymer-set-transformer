#!/usr/bin/env python
# coding=utf-8
#python train.py --training_data "datasets/electrochromics_in_house_experiments_with_abs.csv" --save_dir checkpoints -n_epochs 10 -lr 0.01 --dropout_ratio 0.15
import numpy as np
import pandas as pd
import argparse
import random
import torch 
from copolymer_set_transformer.copolymer_set_transformer import *
from copolymer_set_transformer.ml_modules import *
from sklearn.preprocessing import MinMaxScaler
from copolymer_set_transformer.monomer_representations import get_train_data_representation_dft
import ast
import os
from scipy.signal import savgol_filter


def apply_savgol_filter(y, window_size, poly_order):
    """
    Apply a Savitzky-Golay filter to a 1D array.

    Parameters:
    y (array_like): The input signal.
    window_size (int): The size of the moving window. Must be odd.
    poly_order (int): The order of the polynomial used to fit the samples.

    Returns:
    numpy.ndarray: The smoothed signal.
    """
    return savgol_filter(y, window_size, poly_order)

def set_seed():
  seed = 0
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)

def get_smoothed_intensity(dataset, window_size=30, poly_order=3):
    df_y = []
    for i in range(dataset.shape[0]):
        y=apply_savgol_filter(ast.literal_eval(dataset.intensity[i]), window_size, poly_order)
        df_y.append(y)
    return df_y

def get_scaled_lab(dataset):
    df_y_external = dataset[['L' ,'a', 'b']]
    scalery = MinMaxScaler().fit(df_y_external.values)
    external_val_lab_scaled = scalery.transform(df_y_external)
    return external_val_lab_scaled

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training_data', help='The csv file with the training smiles pairs')
    parser.add_argument('--dropout_ratio', help='Set the dropout_ratio')
    parser.add_argument('--save_dir', help='The directory to save the trained network')
    parser.add_argument('--model_name', help='Name of the trained network')
    parser.add_argument('-n_epochs', default= 50, type = int, help='Set the number of epochs')
    parser.add_argument('-lr', default= 0.001, type = float, help='Set the number of epochs')
    parser.add_argument('-batch_size', default= 32, type = int, help='Set the batch size')
    parser.add_argument('--use_abs_decoder', action='store_true' ,help='Use the absorption decoder')
    args = parser.parse_args()

    # Load the training data
    electrochromics = pd.read_csv(args.training_data)
    electrochromics['smiles1'] = electrochromics['smiles1'].str.replace('*', 'C')
    electrochromics['smiles2'] = electrochromics['smiles2'].str.replace('*', 'C')
    electrochromics['smiles3'] = electrochromics['smiles3'].str.replace('*', 'C')
    df = electrochromics[['smiles1', 'percentage_1', 'smiles2' , 'percentage_2', 'smiles3', 'percentage_3', 'L', 'a', 'b', 'wavelength', 'intensity']]
    df['percentage_1'] = df['percentage_1']/100
    df['percentage_2'] = df['percentage_2']/100
    df['percentage_3'] = df['percentage_3']/100
    df_external = df.dropna(axis=0)
    training_dataset = get_train_data_representation_dft(df_external)
    Ndims = int(training_dataset.shape[1]/3)

    device ='cpu'
    model = CoPolymerSetTransformer(0.15, device, args.n_epochs, args.lr, args.batch_size, use_abs_decoder=False)
    
    # Prepare your data
    train_data_1, train_data_2, train_data_3= np.array(training_dataset.iloc[:, :Ndims].values, dtype=float), np.array(training_dataset.iloc[:, Ndims:2*Ndims].values, dtype=float), np.array(training_dataset.iloc[:, 2*Ndims:].values, dtype=float)
    y_lab = np.array(get_scaled_lab(electrochromics), dtype=np.float16)
    y_abs = np.array(get_smoothed_intensity(electrochromics), dtype=np.float16)
    # Train the model
    losses1, losses2 = model.train_model(train_data_1, train_data_2,train_data_3, y_lab, y_abs)

    
    os.makedirs(args.save_dir, exist_ok=True)
    torch.save({'state_dict':model.state_dict()},
        os.path.join(args.save_dir, 'co_polymer_set_transformer_new_model.tar'))
if __name__ == "__main__":
   main()