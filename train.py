import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import numpy as np 

from util import hot_dog_data, collate_fn
import util
from model import hd_pred

def main(args):
    train_dataset = hot_dog_data(data_path=args.src_file)
    train_loader = data.DataLoader(train_dataset,
                                   shuffle=True, 
                                   batch_size=args.batch_size,
                                   collate_fn=collate_fn)

    model = hd_pred(args)
    model.train()
    optimizer = optim.Adam(params=model.parameters(), lr=args.learning_rate)
    for epoch in range(args.num_epochs):
        print(f"training epoch {epoch}...")
        for i1, i2, i3, i4, i5, i6, i7, i8, i9, i10 in train_loader:
            res = model.forward(i1, i2, i3, i4, i5, i6, i7, i8, i9)
            loss = nn.L1Loss()
            loss_val = loss(res, i10)
            loss_val.backward()
            optimizer.step()

    # store raw data into a csv
    return None


if __name__ == "__main__":
    args = util.get_args()
    main(args)