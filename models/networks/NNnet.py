import torch
from torch import nn
from torch import Tensor
from omegaconf import DictConfig
from torch_geometric.nn import Sequential

from collections import OrderedDict
from torch.nn import BatchNorm1d
from torch.nn import Sequential as Seq, Linear, ReLU, ELU, LeakyReLU
from fairseq.modules import LayerNorm
from ..modules.mappers import SimpleMapper


class NN_module(nn.Module):
    def __init__(self, in_feat, out_feat, **args):
        super(NN_module, self).__init__()
        GCN_mapper = {'SimpleMapper':SimpleMapper}
        self.mapper = GCN_mapper.get(args['GCNEG_mapper_type'])(in_feat, out_feat, cfg=args['cfg']) 
        

    def forward(self, x: Tensor):
        x = self.mapper(x)
        return x

class NNnet(nn.Module):
    """NNnet"""
    def __init__(self, cfg: DictConfig):
        """full cfg.model"""
        super().__init__()

        dynamic_edge_type = {"NN_module": NN_module,}
        
        self.cfg = cfg.model
        self.cfgDataloader = cfg.dataloader

        self.module = dynamic_edge_type.get(self.cfg.edge_generation_type)
        
        if self.module is None:
           raise Exception('self.module is None, but have to be DynamicEdgeConv or DynamicEdgeConv_DCGv')

        # Build GCNnetwork
        # In size and out size depends on the dataset 
        self.cfg.in_channels =  list(self.cfg.in_channels)
      
        # Mapping head
        self.cfg.in_channels = [self.cfg.insize] + list(self.cfg.in_channels) # by the construction

        self.input_bn = BatchNorm1d(self.cfg.insize)

        
        # Build the model 
        names = []
        modules = []
        # Args to pass into layer
        if self.cfg.input_bn == True:
            names.append(f'input_bn')
            modules.append((self.input_bn, 'x -> x'))

        in_feat = self.cfg.in_channels[0]
        out_feat = self.cfg.in_channels[1]       
                  
        act_type = {"RELU":ReLU,
                    "ELU":ELU,
                    "LeakyReLU":LeakyReLU}

        norm_type = {"BN":BatchNorm1d,
                        "LN":LayerNorm,
                        'None': None}

        act = act_type.get(self.cfg.initACT)
        norm = norm_type.get(self.cfg.initNORM)    

        names.append(f'input_mapper')

        modules.append((Seq(Linear(in_feat, out_feat),
                        norm(out_feat),
                        act(),
                        torch.nn.Dropout(p=self.cfg.drop_prob),
                        Linear(out_feat, out_feat),
                        torch.nn.Dropout(p=self.cfg.drop_prob)),
                        'x -> x'))

        in_feat = out_feat

        names.append(f'act')
        modules.append(nn.ReLU())
        

        # Classification head
        self.head_norm = LayerNorm(self.cfg.in_channels[-1], export=False)
        self.classification_head = torch.nn.Linear(self.cfg.in_channels[-1], self.cfg.outsize)
        
        self.model = Sequential('x', OrderedDict(zip(names, modules)))
        
        
        # Reconstruc numerical variables 
        self.num_head = torch.nn.Linear(self.cfg.in_channels[-1], 
                                        len(self.cfgDataloader.imputation.num_idx))
        # Reconstruct cat variables
        self.cat_heads = [torch.nn.Linear(self.cfg.in_channels[-1], dim).to(f'cuda:{cfg.trainer.cuda_number}') \
                            for dim in self.cfgDataloader.imputation.cat_dims]
            
    def forward(self, x):
        output = self.model(x=x)
        output_norm = self.head_norm(output)

        logits = self.classification_head(output_norm)
        
        num_rec = self.num_head(output_norm)

        cat_outputs = []
        for head in self.cat_heads:
            cat_outputs.append(head(output_norm))
    
        return logits, num_rec, cat_outputs

        
