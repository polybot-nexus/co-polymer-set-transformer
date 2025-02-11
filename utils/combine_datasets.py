from colour.plotting import *
import colour
import matplotlib.pyplot as plt
import pandas as pd
import csv
import os

# df1 = pd.read_csv('C:/Users/kvriz/Desktop/Polybot_ECPs/datasets/electrochromics_literature_only.csv')
df1 = pd.read_csv('C:/Users/kvriz/Desktop/Polybot_ECPs/datasets/electrochromics_in_house_experiments.csv')

df2 = pd.read_csv('C:/Users/kvriz/Desktop/Polybot_ECPs/datasets/absorption_spectra/all_abs_spectra_corrected.csv')
# print(df1.polymer_id.values)
# print(len(df1.polymer_id.values))
# Merging datasets on 'Citation' and 'Polymer Id'
merged_data = pd.merge(df1, df2[['source', 'polymer_id', 'wavelength', 'intensity']], on=['source', 'polymer_id'], how='inner')
# print(merged_data)
# merged_data.to_csv('literature_only_dataset_with_abs.csv', index=None)
merged_data.to_csv('electrochromics_in_house_experiments_with_abs.csv', index=None)
# merged_data = pd.merge(df1, df2[['source', 'polymer_id', 'wavelength', 'intensity']], on=['source', 'polymer_id'], 
#                        how='outer', 
#                        indicator=True)

# Filter the merged data to find rows that only exist in one of the datasets
# non_matches = merged_data[merged_data['_merge'] != 'both']

# Print out non-matches
# print(non_matches)

# If you want to examine non-matching entries from df1 and df2 separately, you can do:
# non_matches_df1 = non_matches[non_matches['_merge'] == 'left_only']
# non_matches_df2 = non_matches[non_matches['_merge'] == 'right_only']

# print("Entries in df1 that didn't find a match in df2:")
# print(non_matches_df1)

# print("\nEntries in df2 that didn't find a match in df1:")
# print(non_matches_df2)