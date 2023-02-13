import torch
from torch import nn, Tensor
from torch.nn import Sequential as Seq, Linear, ReLU
from torch.nn import BatchNorm1d

class SimpleMapper(nn.Module):
    def __init__(self, in_feat, out_feat, **args):
        super(SimpleMapper, self).__init__()
        self.prob = args['cfg'].mapperDP
    
        act = ReLU()
        norm = BatchNorm1d(out_feat)
        norm2 = BatchNorm1d(out_feat)
        
        self.mapper = Seq(
                        Linear(in_feat, out_feat),
                        norm,
                        act, 
                        torch.nn.Dropout(p=self.prob),
                        Linear(out_feat, out_feat),
                        norm2,
                        act, 
                        )
    
    def forward(self, x: Tensor):
        x = self.mapper(x)
        return x