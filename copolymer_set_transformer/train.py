#!/usr/bin/env python
# coding=utf-8
import sys
import json
import tempfile
import numpy as np
import pandas as pd
import os.path
from collections import Counter
from itertools import groupby
import argparse
import random
import torch 
from rdkit import Chem
from rdkit.Chem import AllChem
from copolymer_set_transformer.copolymer_set_transformer import *
from copolymer_set_transformer.ml_modules import *
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional
import datasets
from datasets import load_dataset



logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """
    Arguments to train the model with user provided data or test on existing datasets.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override some existing default config settings when a model is trained from scratch. Example: "
            "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )

    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default= 'morgan', choices = ['gnn', 'morgan', 'chemberta', 'mordred'] , help='The model to use for training')
    parser.add_argument('--training_data', help='The csv file with the training smiles pairs')
    parser.add_argument('--save_dir', help='The directory to save the trained network')
    parser.add_argument('--model_name', help='Name of the trained network')
    parser.add_argument('-n_epochs', default= 50, type = int, help='Set the number of epochs')
    parser.add_argument('-lr', default= 0.001, type = float, help='Set the learning rate')
    parser.add_argument('--use_wandb', action='store_true' ,help='Set the learning rate')
    args = parser.parse_args()

    #set_seed()
   
#if __name__ == "__main__":
#    main()