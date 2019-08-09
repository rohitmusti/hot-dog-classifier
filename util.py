import torch
import numpy as np 
import pandas as pd 

import argparse

def get_args():
    parser = argparse.ArgumentParser('Hot Dog Trainer')
    parser.add_argument("--src_file",
                        type=str,
                        required=True,
                        default="./hot_dog.csv",
                        help="file from which to extract the hot dog information")
    parser.add_argument("--num_epochs",
                        type=int,
                        required=True,
                        default=200,
                        help="number of epochs to train model for")
    args = parser.parse_args()
    return args

class qcd(data.Dataset):
    def __init__(self, data_path):

        with open(args.src_file, "r") as fh:
            src = np.loadtxt(fh, delimiter=',', skiprows=1)

        

        # features
        self.mass = torch.tensor(5)
        self.calories = torch.tensor(5)
        self.fat = torch.tensor(5)
        self.cholestrol = torch.tensor(5)
        self.sodium = torch.tensor(5)
        self.potassium = torch.tensor(5)
        self.carbohydrates = torch.tensor(5)
        self.protein = torch.tensor(5)

        # label
        self.time


    def __getitem__(self, idx):

        examples = (
            self.mass[idx],
            self.calories[idx],
            self.fat[idx],
            self.cholestrol[idx]
            self.sodium,
            self.potassium,
            self.carbohydrates,
            self.protein,
            self.time[idx],
            self.temp[idx]
            

        )

    
    def __len__(self):
        return len(self.ids)