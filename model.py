import torch
import torch.nn as nn

class hd_pred(nn.Module):
    def __init__(self, args):
        super(hd_pred, self).__init__()

        self.l1 = nn.Linear(in_features=9, out_features=args.hidden_size)
        self.l2 = nn.Linear(in_features=args.hidden_size, out_features=args.hidden_size)
        self.l3 = nn.Linear(in_features=args.hidden_size, out_features=args.hidden_size*3)
        self.l4 = nn.Linear(in_features=args.hidden_size*3, out_features=args.hidden_size*4)
        self.l5 = nn.Linear(in_features=args.hidden_size*4, out_features=args.hidden_size)
        self.l6 = nn.Linear(in_features=args.hidden_size, out_features=args.hidden_size)
        self.l7 = nn.Linear(in_features=args.hidden_size, out_features=args.hidden_size)
        self.l8 = nn.Linear(in_features=args.hidden_size, out_features=1)
        self.sig = nn.Sigmoid()
        self.out = nn.ReLU()

    def forward(self, f1, f2, f3, f4, f5, f6, f7, f8, f9):
        combined=[]
        for i in range(int(f1.size()[0])):
            combined.append(torch.tensor([f1[i], f2[i], f3[i], f4[i], f5[i], f6[i], f7[i], f8[i], f9[i]]))
        
        combined = torch.stack(combined)
        l1_out = self.l1(combined)
        out = self.out(l1_out)
        # l2_out = self.l2(out)
        # out = self.out(l2_out)
        # l3_out = self.l3(out)
        # out = self.out(l3_out)
        # l4_out = self.l4(out)
        # out = self.out(l4_out)
        # l5_out = self.l5(out)
        # out = self.out(l5_out)
        # l6_out = self.l6(out)
        # out = self.out(l6_out)
        # l7_out = self.l7(out)
        # out = self.out(l7_out)
        l8_out = self.l8(out)
        out = self.out(l8_out)

        return out