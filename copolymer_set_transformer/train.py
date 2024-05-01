#!/usr/bin/env python
# coding=utf-8
import numpy as np
import pandas as pd
from collections import Counter
from itertools import groupby
import argparse
import random
import torch 
from copolymer_set_transformer.copolymer_set_transformer import *
from copolymer_set_transformer.ml_modules import *
from sklearn.preprocessing import MinMaxScaler
from itertools import chain
from typing import Optional
from monomer_representations import get_train_data_representation_dft
import ast
from scipy.signal import find_peaks, find_peaks_cwt
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
    df_y_external = dataset[['L* (Colored State)' ,'a* (Colored State)', 'b*(Colored State)']]
    scalery = MinMaxScaler().fit(df_y_external.values)
    external_val_lab_scaled = scalery.transform(df_y_external)
    return external_val_lab_scaled

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default= 'morgan', choices = ['gnn', 'morgan', 'chemberta', 'mordred'] , help='The model to use for training')
    parser.add_argument('--training_data', help='The csv file with the training smiles pairs')
    parser.add_argument('--dropout_ratio', help='Set the dropout_ratio')
    parser.add_argument('--save_dir', help='The directory to save the trained network')
    parser.add_argument('--model_name', help='Name of the trained network')
    parser.add_argument('-n_epochs', default= 50, type = int, help='Set the number of epochs')
    parser.add_argument('-n_epochs', default= 50, type = int, help='Set the number of epochs')
    parser.add_argument('-batch_size', default= 0.001, type = float, help='Set the batch size')
    parser.add_argument('--use_abs_decoder', action='store_true' ,help='Use the absorption decoder')
    args = parser.parse_args()

    # Load the training data
    electrochromics = args.training_data
    electrochromics['smiles_A'] = electrochromics['smiles_A'].str.replace('*', 'C')
    electrochromics['smiles_B'] = electrochromics['smiles_B'].str.replace('*', 'C')
    electrochromics['smiles_C'] = electrochromics['smiles_C'].str.replace('*', 'C')
    df = electrochromics[['smiles_A', 'Percentage of A %', 'smiles_B' , 'Percentage of B %', 'smiles_C', 'Percentage of C %', 'L* (Colored State)', 'a* (Colored State)', 'b*(Colored State)', 'wavelength', 'intensity']]
    df['Percentage of A %'] = df['Percentage of A %']/100
    df['Percentage of B %'] = df['Percentage of B %']/100
    df['Percentage of C %'] = df['Percentage of C %']/100
    df_external = df.dropna(axis=0)
    training_dataset = get_train_data_representation_dft(df_external)
    Ndims = int(training_dataset.shape[1]/3)

    device ='cpu'
    model = CoPolymerSetTransformer(args.dropout_ratio, device, args.n_epochs, args.lr, args.batch_size, use_abs_decoder=False)
    # Prepare your data
    train_data_1, train_data_2, train_data_3= np.array(training_dataset.iloc[:, :Ndims].values, dtype=float), np.array(training_dataset.iloc[:, Ndims:2*Ndims].values, dtype=float), np.array(training_dataset.iloc[:, 2*Ndims:].values, dtype=float)
    y_lab = np.array(get_scaled_lab(electrochromics), dtype=np.float16)
    y_abs = np.array(get_smoothed_intensity(electrochromics), dtype=np.float16)
    # Train the model
    losses1, losses2 = model.train_model(train_data_1, train_data_2,train_data_3, y_lab, y_abs)
    model._save_to_state_dict(args.save_dir)

if __name__ == "__main__":
   main()