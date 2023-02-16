import torch
from torch import nn
from torch.nn import BatchNorm1d
from torch.nn import Sequential as Seq, Linear, ReLU, ELU, LeakyReLU
from torch.nn import LayerNorm
from torch_geometric.nn import Sequential

from omegaconf import DictConfig
from collections import OrderedDict

from ..modules.egg import EGG_module, kEGG_GAE

import numpy as np
class EGnet(nn.Module):
    """EGnet"""
    def __init__(self, cfg: DictConfig):
        """full cfg.model"""
        super().__init__()

        dynamic_edge_type = {
                            "EGG_module":EGG_module,
                            "kEGG_GAE":kEGG_GAE
                            }
        
        self.cfg = cfg.model
        self.cfgDataloader = cfg.dataloader
        self.module = dynamic_edge_type.get(self.cfg.edge_generation_type)
        
        if self.module is None:
           raise Exception('self.module is None,')

        # Build GCNnetwork
        # In size and out size depends on the dataset 
        self.cfg.in_channels =  list(self.cfg.in_channels)
      
        assert len(self.cfg.GCNEG_head.types) == len(self.cfg.in_channels), f'{len(self.cfg.GCNEG_head.types)} != {len(self.cfg.in_channels)}'
        assert len(self.cfg.GumbleDistFunc.types) == len(self.cfg.in_channels), f'{len(self.cfg.GumbleDistFunc.types)} != {len(self.cfg.in_channels)}'
        
        if self.cfg.edge_generation_type == 'kEGG_GAE':
            self.cfg.k_degree = list(self.cfg.k_degree) * len(self.cfg.GCNEG_head.types)        
        
        # Mapping head
        self.cfg.in_channels = [self.cfg.insize] + list(self.cfg.in_channels)

        self.input_bn = BatchNorm1d(self.cfg.insize)
        
        # Build the model 
        names = []
        modules = []
        
        args_names = ['GCNEG_head_type', 'GCNEG_mapper_type','GumbleDistFunc_type', 'SamplingProcedure_type', 'k_degree']
        for idx, values in enumerate(zip(self.cfg.GCNEG_head.types,
                                         self.cfg.GCNEG_mapper.types,
                                         self.cfg.GumbleDistFunc.types,
                                         self.cfg.SamplingProcedure.types, 
                                         self.cfg.k_degree)):
                                 
            args = dict(zip(args_names, values))
            in_feat = self.cfg.in_channels[idx]
            out_feat = self.cfg.in_channels[idx+1]

            edge_out_feat =  self.cfg.edge_out_feat[idx]

            if idx == 0:          
                act_type = {"RELU":ReLU,
                            "ELU":ELU,
                            "LeakyReLU":LeakyReLU}

                norm_type = {"BN":BatchNorm1d,
                             "LN":LayerNorm,
                             'None': None}

                act = act_type.get(self.cfg.initACT)
                norm1 = norm_type.get(self.cfg.initNORM)    
                norm2 = norm_type.get(self.cfg.initNORM)    

                self.input_mapper = Seq(Linear(in_feat, out_feat),
                                        norm1(out_feat),
                                        act(),
                                        torch.nn.Dropout(p=self.cfg.drop_prob),
                                        Linear(out_feat, out_feat),
                                        norm2(out_feat))

                in_feat = out_feat
           
            
            names.append(f'EGG_{idx}')
            modules.append((self.module(in_feat=in_feat, out_feat=out_feat, edge_out_feat=edge_out_feat,
                            cfg=self.cfg,
                            **args), 'x -> x')
                          )
            
            names.append(f'act')
            modules.append( (nn.LeakyReLU()) )
            
        self.egg_modules = Sequential('x', OrderedDict(zip(names, modules)))
        
        # Classification head
        self.head_norm = LayerNorm(self.cfg.in_channels[-1])
        
        self.classification_head = torch.nn.Linear(self.cfg.in_channels[-1], self.cfg.outsize)
        
        # Reconstruc numerical variables 
        self.num_head = torch.nn.Linear(self.cfg.in_channels[-1], 
                                        len(self.cfgDataloader.imputation.num_idx))
        # Reconstruct cat variables
        self.cat_heads = torch.nn.ModuleList([torch.nn.Linear(self.cfg.in_channels[-1], dim).to(f'cuda:{cfg.trainer.cuda_number}') \
                            for dim in self.cfgDataloader.imputation.cat_dims])
        

    def forward(self, x):
        x = self.input_bn(x)
        x = self.input_mapper(x)

        
        emd = self.egg_modules(x)
        emd_norm = self.head_norm(emd)

        logits = self.classification_head(emd_norm)
        
        num_rec = self.num_head(emd_norm)

        cat_outputs = []
        for head in self.cat_heads:
            cat_outputs.append(head(emd_norm))
        
        return logits, num_rec, cat_outputs
