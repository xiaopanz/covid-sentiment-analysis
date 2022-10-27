import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class CNN(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        D = 50
        C = 2
        Ci = 1
        Co = 200
        Ks = [2, 3, 4, 5]

        self.convs = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(len(Ks) * Co, Co)
        self.fc2 = nn.Linear(Co, C)

    def forward(self, x):
        #(128, max_sent, 50)
        x = x.unsqueeze(1)  # (N, Ci, W, D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)

        x = self.dropout(x)  # (N, len(Ks)*Co)
        x = self.fc1(x)  # (N, C)
        logit = self.fc2(x)
        return logit