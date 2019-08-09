import torch
import torch.nn as nn

class hd_pred(nn.Module):
    def __init__(self, args):
        super(hd_pred, self).__init__()

        self.l1 = nn.Linear(in_features=9, out_features=args.hidden_size)
        self.l2 = nn.Linear(in_features=args.hidden_size, out_features=args.hidden_size*2)
        self.l3 = nn.Linear(in_features=args.hidden_size*2, out_features=args.hidden_size*3)
        self.l4 = nn.Linear(in_features=args.hidden_size*3, out_features=args.hidden_size*4)
        self.l5 = nn.Linear(in_features=args.hidden_size*4, out_features=args.hidden_size)
        self.l6 = nn.Linear(in_features=args.hidden_size, out_features=args.hidden_size)
        self.l7 = nn.Linear(in_features=args.hidden_size, out_features=args.hidden_size)
        self.l8 = nn.Linear(in_features=args.hidden_size, out_features=1)
        self.out = nn.ReLU()

    def forward(self, f1, f2, f3, f4, f5, f6, f7, f8, f9):
        combined=[]
        for i in range(int(f1.size()[0])):
            combined.append(torch.tensor([f1[i], f2[i], f3[i], f4[i], f5[i], f6[i], f7[i], f8[i], f9[i]]))
        
        combined = torch.stack(combined)
        l1_out = self.l1(combined)
        l2_out = self.l2(l1_out)
        l3_out = self.l3(l2_out)
        l4_out = self.l4(l3_out)
        l5_out = self.l5(l4_out)
        l6_out = self.l6(l5_out)
        l7_out = self.l7(l6_out)
        l8_out = self.l8(l7_out)
        out = self.out(l8_out)

        return out


        return 0