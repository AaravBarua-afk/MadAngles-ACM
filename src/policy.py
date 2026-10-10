"""Exact accepted V13 Policy class; no training code."""
import torch
from torch import nn

class Policy(nn.Module):
    def __init__(self,kind='B'):
        super().__init__();self.kind='B'
        self.encoder=nn.Sequential(nn.Linear(13,64),nn.ReLU())
        self.gru=nn.GRU(64,64,num_layers=1,batch_first=True,bidirectional=False)
        self.head=nn.Linear(64,5)
    def forward(self,x,h=None):
        z,h=self.gru(self.encoder(x).unsqueeze(0),h)
        out=self.head(z.squeeze(0))
        return out[:,:4],torch.tanh(out[:,4]),h
