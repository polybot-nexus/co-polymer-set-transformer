# importing necessary libraries
import pandas as pd
import numpy as np
import torch
import tqdm
import torch.nn as nn
from copolymer_set_transformer.ml_modules import SAB, PMA
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
device = 'cpu'

# Helper functions
def combine(data1, data2):
    return np.array(list(zip(data1, data2)))

def enable_dropout(m):
  for each_module in m.modules():
    if each_module.__class__.__name__.startswith('Dropout'):
      each_module.train()

def weighted_mse_loss(input, target):
    weights = Variable(torch.Tensor([0.5,1,1]))#.cuda()  # Higher weight for the a and b values than for the L
    pct_var = (input-target)**2
    out = pct_var * weights.expand_as(target)
    loss = out.mean() 
    return loss

###########################################################################################
class MonomerPair(torch.utils.data.Dataset):
  def __init__(self, dataset1, dataset2, dataset3, y):
    self.x = np.array(list(zip(dataset1, dataset2, dataset3)))
    self.x = np.concatenate([dataset1, dataset2, dataset3], axis=-1)
    self.y = np.reshape(y, (y.shape[0], -1))
    
  def __getitem__(self, index):
    return self.x[index], self.y[index]
  
  def __len__(self):
    return len(self.x)
  
###########################################################################################

class CoPolymerSetTransformer(nn.Module):
    def __init__(self, dropout_ratio, device, epochs, learning_rate, batch_size, use_abs_decoder=False):

        super().__init__()
        self.dropout_ratio = dropout_ratio
        self.device = device
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.use_abs_decoder = use_abs_decoder  # Setting the flag as a class attribute

        self.enc = nn.Sequential(
            SAB(dim_in=1056, dim_out=800, num_heads=5),
            nn.Dropout(p=self.dropout_ratio),
            nn.LayerNorm(800),
            SAB(dim_in=800, dim_out=500, num_heads=5),
            nn.Dropout(p=self.dropout_ratio),
            nn.LayerNorm(500)
        )

        self.dec1 = nn.Sequential(
            PMA(dim=500, num_heads=4, num_seeds=1),
            SAB(dim_in=500, dim_out=200, num_heads=5),
            nn.Dropout(p=self.dropout_ratio),
            nn.LayerNorm(200),
            SAB(dim_in=200, dim_out=100, num_heads=5),
            nn.LayerNorm(100),
            nn.Dropout(p=self.dropout_ratio),
            nn.LeakyReLU(),
            nn.Linear(in_features=100, out_features=3)
        )

        if self.use_abs_decoder:
            self.dec2 = nn.Sequential(
                PMA(dim=500, num_heads=2, num_seeds=1),
                SAB(dim_in=500, dim_out=400, num_heads=2),
                nn.LayerNorm(400),
                nn.Dropout(p=self.dropout_ratio),
                nn.LeakyReLU(),
                nn.Linear(in_features=400, out_features=451)
            )


    def forward(self, x):
        x = x.reshape((x.shape[0], 3, -1)).float() 
        encoded_features = self.enc(x)
        output1 = self.dec1(encoded_features)
        if self.use_abs_decoder:
          output2 = self.dec2(encoded_features)
          return output1.squeeze(-1).squeeze(1), output2.squeeze(-1).squeeze(1)
        return output1.squeeze(-1).squeeze(1)


    def train_model(self, data1, data2, data3, y_lab=None, y_abs=None):
        self.to(self.device)
        torch.manual_seed(0)
        enable_dropout(self)

        # Optimizers for each decoder
        optimizer1 = torch.optim.Adam(self.dec1.parameters(), lr=self.learning_rate)
        criterion1 = weighted_mse_loss
        losses1 = []

        if self.use_abs_decoder:
            optimizer2 = torch.optim.Adam(self.dec2.parameters(), lr=self.learning_rate)
            criterion2 = nn.MSELoss()
            losses2 = []     
        
        for _ in tqdm.tqdm(range(self.epochs)):
            #if y_lab is not None:
            for x, y in DataLoader(MonomerPair(data1, data2, data3, y_lab), batch_size=self.batch_size):
                x, y = x.to(self.device).float(), y.to(self.device).float()
                optimizer1.zero_grad()
                output1 = self(x)

                if self.use_abs_decoder:
                    loss1 = criterion1(output1[0], y)
                else:
                  loss1 = criterion1(output1, y)             
               
                loss1.backward()
                optimizer1.step()
                losses1.append(loss1.item())

            if self.use_abs_decoder:
                for x, y in DataLoader(MonomerPair(data1, data2, data3, y_abs), batch_size=self.batch_size):
                    x, y = x.to(self.device).float(), y.to(self.device).float()
                    optimizer2.zero_grad()
                    output2 = self(x)
                    loss2 = criterion2(output2[1], y)
                    loss2.backward()
                    optimizer2.step()
                    losses2.append(loss2.item())
        if self.use_abs_decoder:
            return losses1, losses2

        return losses1, None


    def test_model(self, data1, data2, data3, target_lab, target_abs=None):
        lab_list=[]
        abs_list=[]
        y_list_std1=[]
        y_list_std2=[]

        torch.manual_seed(0)
        enable_dropout(self)

        for i in range(10):
            self = self.to(self.device)
            for x, y in DataLoader(MonomerPair(data1, data2, data3, target_lab), batch_size=len(data1)):
                x, y = x.float(), y.float() #.to(self.device)
                y= self(x)
                lab_list.append(y.detach().numpy())
                y_list_std1.append(y[:, :].detach().numpy())


            if self.use_abs_decoder:
              for x, y in DataLoader(MonomerPair(data1, data2, data3, target_abs), batch_size=len(data1)):
                  x, y = x.float(), y.float() #.to(self.device)
                  y= self(x)
                  abs_list.append(y.detach().numpy())
                  y_list_std2.append(y[:, :].detach().numpy())
        
              overal_std = np.std((abs_list , lab_list),axis=0)
              return np.mean(abs_list, axis=0), np.mean(lab_list, axis=0), overal_std 

        return np.mean(lab_list, axis=0), np.std(np.std(y_list_std1, axis=0), axis=1)