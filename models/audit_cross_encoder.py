"""Import-safe OD regression heads with explicit symmetric/asymmetric dimensions.
All CrossEncoder heads can yield asymmetric OD predictions because inputs are
ordered pairs. No shared-node metric or triangle guarantee is claimed here.
"""
import torch
from torch import nn
from utils.asymmetric_metrics import L1Tilde

class CrossEncoder(nn.Module):
    def __init__(self,d_in,hidden=512,output_dim=64,mode='l1',r=62,s=2):
        super().__init__()
        if mode not in ('l1','l1tilde','scalar'):
            raise ValueError(mode)
        if r+s != output_dim or min(r,s)<0:
            raise ValueError('Require r+s=output_dim and r,s>=0')
        self.mode,self.r,self.s = mode,r,s
        self.output_dim=output_dim
        self.metric=L1Tilde(r,s)
        self.backbone=nn.Sequential(
            nn.Linear(d_in,hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(.1),
            nn.Linear(hidden,hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(.1))
        self.head=nn.Linear(hidden,1 if mode=='scalar' else 2*output_dim)

    def decode(self,out):
        if self.mode=='scalar':
            return torch.nn.functional.softplus(out)
        x,y=out[:,:self.output_dim],out[:,self.output_dim:]
        if self.mode=='l1':
            return torch.abs(y-x).mean(dim=1,keepdim=True)
        return self.metric(x,y)/self.output_dim

    def forward(self,x):
        return self.decode(self.head(self.backbone(x)))
