import torch
import torch.utils.data as data
import torch.nn.functional as F
import torch.nn as nn
import numpy as np 
import pandas as pd 
import argparse

def get_args():
    parser = argparse.ArgumentParser('Hot Dog Trainer')
    parser.add_argument("--src_file",
                        type=str,
                        required=False,
                        default="./hot_dog.csv",
                        help="file from which to extract the hot dog information")
    parser.add_argument("--batch_size",
                        type=int,
                        required=False,
                        default=2,
                        help="number of epochs to train model for")
    parser.add_argument("--learning_rate",
                        type=float,
                        required=False,
                        default=.002,
                        help="number of epochs to train model for")
    parser.add_argument("--num_epochs",
                        type=int,
                        required=False,
                        default=200,
                        help="number of epochs to train model for")
    parser.add_argument("--hidden_size",
                        type=int,
                        required=False,
                        default=100,
                        help="hidden size of layers in neural net")
    args = parser.parse_args()
    return args

class hot_dog_data(data.Dataset):
    def __init__(self, data_path):

        # features
        self.ids = []
        self.mass = []
        self.calories = []
        self.fat = []
        self.cholestrol = []
        self.sodium = []
        self.potassium = []
        self.carbohydrates = []
        self.protein = []
        self.time = []

        # temp
        self.temp = []


        with open(data_path, "r") as fh:
            src = np.loadtxt(fh, delimiter=',', skiprows=1)

        for row in src:
            self.ids.append(row[0])
            self.mass.append(row[1])
            self.calories.append(row[2])
            self.fat.append(row[3])
            self.cholestrol.append(row[4])
            self.sodium.append(row[5])
            self.potassium.append(row[6])
            self.carbohydrates.append(row[7])
            self.protein.append(row[8])
            self.time.append(row[9])
            self.temp.append(row[10])

        self.ids = torch.tensor(self.ids)
        self.mass = F.normalize(torch.tensor(self.mass), dim=0)
        self.calories = F.normalize(torch.tensor(self.calories), dim=0)
        self.fat = F.normalize(torch.tensor(self.fat), dim=0)
        self.cholestrol = F.normalize(torch.tensor(self.cholestrol), dim=0)
        self.sodium = F.normalize(torch.tensor(self.sodium), dim=0)
        self.potassium = F.normalize(torch.tensor(self.potassium), dim=0)
        self.carbohydrates = F.normalize(torch.tensor(self.carbohydrates), dim=0)
        self.protein = F.normalize(torch.tensor(self.protein), dim=0)
        self.time = F.normalize(torch.tensor(self.time), dim=0)
        self.temp = F.normalize(torch.tensor(self.temp), dim=0)


    def __getitem__(self, idx):
        # idx = self.ids[idx]
        examples = (
            self.mass[idx],
            self.calories[idx],
            self.fat[idx],
            self.cholestrol[idx],
            self.sodium[idx],
            self.potassium[idx],
            self.carbohydrates[idx],
            self.protein[idx],
            self.time[idx],
            self.temp[idx]
        )

        return examples

    
    def __len__(self):
        return len(self.ids)

def collate_fn(examples):
    """
    Adapted from:
        https://github.com/yunjey/seq2seq-dataloader
    """
    def merge_0d(scalars, dtype=torch.float):
        return torch.tensor(scalars, dtype=dtype)

    mass,calories,fat,cholestrol,sodium,potassium,carbohydrates,protein,time,temp = zip(*examples)

    mass = merge_0d(mass)
    calories = merge_0d(calories)
    fat = merge_0d(fat)
    cholestrol = merge_0d(cholestrol)
    sodium = merge_0d(sodium)
    potassium = merge_0d(potassium)
    carbohydrates = merge_0d(carbohydrates)
    protein = merge_0d(protein)
    time = merge_0d(time)
    temp = torch.squeeze(merge_0d(temp))

    return (mass, calories, fat, cholestrol, sodium, potassium, carbohydrates, protein, time, temp)