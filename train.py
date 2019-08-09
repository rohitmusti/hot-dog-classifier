import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import numpy as np 
from tqdm import tqdm

from util import hot_dog_data, collate_fn
import util
from model import hd_pred

def main(args):
    best_loss = 0
    best_hs = 0
    best_e = 0
    best_batch_size = 0
    for hs in range(2,100):
        args.hidden_size = hs
        for bs in range(2,10):
            args.batch_size = bs
            for e in range(2, 200):
                args.num_epochs = e
                torch.manual_seed(3716)
                train_dataset = hot_dog_data(data_path=args.src_file)
                train_loader = data.DataLoader(train_dataset,
                                               shuffle=True, 
                                               batch_size=args.batch_size,
                                               collate_fn=collate_fn)

                model = hd_pred(args)
                model.train()
                optimizer = optim.Adam(params=model.parameters(), lr=args.learning_rate)
                loss = nn.MSELoss()
                for _ in tqdm(range(args.num_epochs)):
                    for i1, i2, i3, i4, i5, i6, i7, i8, i9, i10 in train_loader:
                        res = model.forward(i1, i2, i3, i4, i5, i6, i7, i8, i9)
                        loss_val = loss(res, i10)
                        loss_val.backward()
                        optimizer.step()


                correct = 0
                for i1, i2, i3, i4, i5, i6, i7, i8, i9, i10 in train_loader:
                    res = model.forward(i1, i2, i3, i4, i5, i6, i7, i8, i9)
                    loss_val = loss(res, i10)
                    if abs(res - i10) < .001:
                        correct += 1

                if correct > best_loss:
                    best_loss = correct
                    best_hs = args.hidden_size
                    best_e = args.num_epochs
                    best_batch_size = args.batch_size


                
    

    print(f"best amount correct: {best_loss}")
    print(f"best hidden_size: {best_hs}")
    print(f"best num_epochs: {best_e}")
    print(f"best batch_size: {best_batch_size}")

    return None


if __name__ == "__main__":
    args = util.get_args()
    main(args)