from torch import nn

from torch import Tensor
from ..modules.mappers import SimpleMapper


class NN_module(nn.Module):
    def __init__(self, in_feat, out_feat, **args):
        super(NN_module, self).__init__()
        GCN_mapper = {'SimpleMapper':SimpleMapper}
        self.mapper = GCN_mapper.get(args['GCNEG_mapper_type'])(in_feat, out_feat, cfg=args['cfg']) 
        

    def forward(self, x: Tensor):
        x = self.mapper(x)
        return x