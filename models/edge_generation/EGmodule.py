from torch import nn
import torch
from torch import Tensor
from .sampler import GumbelSigmoid, GumbelSigmoid_k


class EdgeGenerationModule(nn.Module):
    def __init__(self, **args):
        super(EdgeGenerationModule, self).__init__()
        SamplingProcedure = {
                        "GumbelSigmoid": GumbelSigmoid,
                        "GumbelSigmoid_k": GumbelSigmoid_k
                        }

        self.cfg = args['cfg'] # """Gets cfg.model"""
        self.gumble = SamplingProcedure.get(args['SamplingProcedure_type'])(**args)
        self.current_step = -1
        
        self.temp = torch.tensor([1], dtype=torch.float32)
        self.tau = torch.tensor([0.5], dtype=torch.float32)
       
    def forward(self, x: Tensor):
       
        self.StatDict = {}
        edge_index, edge_weight,\
        self.StatDict['probs'],\
        self.StatDict['mask'],\
        self.StatDict['dist_mat'] = self.gumble.forward(x=x, 
                                                tau=self.tau.to(x.device),
                                                temperature=self.temp.to(x.device))
        
        return edge_index, edge_weight

