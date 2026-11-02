"""Pairwise model: individual-invariant severity plus signed inter-eye laterality."""
import torch
import torch.nn as nn
from config import DROPOUT

class Block(nn.Module):
    def __init__(self, i, o, dropout=DROPOUT):
        super().__init__(); self.net=nn.Sequential(nn.Linear(i,o),nn.LayerNorm(o),nn.GELU(),nn.Dropout(dropout))
    def forward(self,x): return self.net(x)

class PairwiseAmblyopiaClassifier(nn.Module):
    """Paired tabular classifier without missingness-mask inputs."""
    def __init__(self, n_bio, dropout=DROPOUT):
        super().__init__(); self.n_bio=n_bio
        self.demo=nn.Sequential(Block(2,24,dropout),nn.Linear(24,16),nn.LayerNorm(16),nn.GELU())
        self.eye=nn.Sequential(Block(n_bio,48,dropout),nn.Linear(48,32),nn.LayerNorm(32),nn.GELU())
        # signed difference, absolute difference, and pair mean
        self.pair=nn.Sequential(Block(3*n_bio,96,dropout),nn.Linear(96,48),nn.LayerNorm(48),nn.GELU())
        self.head=nn.Sequential(Block(16+32+48+32,64,dropout),nn.Linear(64,24),nn.LayerNorm(24),nn.GELU())
        self.severity=nn.Linear(24,1)
        self.laterality=nn.Linear(24,1)
    def forward(self,x,m=None):
        n=self.n_bio; demo=self.demo(x[:,:2])
        r,l=x[:,2:2+n],x[:,2+n:2+2*n]
        er=self.eye(r); el=self.eye(l)
        diff=r-l; absolute=torch.abs(diff); mean=(r+l)/2
        pair=self.pair(torch.cat([diff,absolute,mean],1))
        fused=self.head(torch.cat([demo,pair,(er+el)/2,torch.abs(er-el)],1))
        base=self.severity(fused).squeeze(1); side=self.laterality(fused).squeeze(1)
        # side is signed and learned from R-L; logits remain explicitly R/L.
        return torch.stack([base+side,base-side],1)
