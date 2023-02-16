from torch import nn
from torch_geometric.nn import GCNConv, GATConv, ARMAConv, SGConv
from torch import Tensor
from ..edge_generation.EGmodule import EdgeGenerationModule
from torch.nn import LayerNorm
from torch_geometric.nn import MessagePassing
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.typing import OptTensor
import torch

class EGG_module(nn.Module):
    def __init__(self, in_feat, out_feat, edge_out_feat, **args):
        super(EGG_module, self).__init__()
        args['in_feat'] = in_feat
        self.edge_gen = EdgeGenerationModule(**args)
        GCNEG_head = {  "GCNConv": GCNConv,
                        "EdgeConv": EdgeConv,
                        "GATConv": GATConv,
                        "ARMAConv":ARMAConv,
                        "SGConv":SGConv,
                        }
                        
        #GCN_mapper = {'SimpleMapper':SimpleMapper}
        
        self.cfg = args['cfg'] # """Gets cfg.model"""
        
        if args['GCNEG_head_type'] == "GCNConv":
            self.gcn = GCNEG_head.get(args['GCNEG_head_type'])(out_feat, out_feat, improved=True)
        else: 
            self.gcn = GCNEG_head.get(args['GCNEG_head_type'])(out_feat, out_feat) 
        
        # self.mapper = GCN_mapper.get(args['GCNEG_mapper_type'])(in_feat, edge_out_feat, cfg=self.cfg)
        self.mapper = Seq(Linear(in_feat, in_feat),
                          ReLU(), 
                          Linear(in_feat, in_feat)
                        ) 

        self.norm=LayerNorm(out_feat)
        

    def forward(self, x: Tensor):
        self.StatDict = {}
        mapped_x = self.mapper(x)
        edge_index, edge_weight = self.edge_gen(mapped_x)
        

        # Apply gcn to proposed edges and initial x
        # x_gcn = self.gcn(self.norm(x), edge_index, edge_weight)
        # x = x_gcn + x
        x = self.gcn(x, edge_index, edge_weight)

        return x

class EdgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels, **args):
        super().__init__(aggr='mean') 
        self.mlp = Seq(Linear(2 * in_channels, out_channels),
                       ReLU(),
                       Linear(out_channels, out_channels))

    def forward(self, x, edge_index, edge_weight: OptTensor = None):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        return self.propagate(edge_index=edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_i, x_j, edge_weight: OptTensor):
        tmp = torch.cat([x_i, x_j - x_i], dim=1)  # tmp has shape [E, 2 * in_channels]
        return self.mlp(tmp) if edge_weight is None else self.mlp(tmp) * edge_weight.view(-1,1)


# kEGG-GAE
class kEGG_GAE(EdgeConv):
    def __init__(self, in_feat, out_feat, cfg, **args):
        super().__init__(in_feat, out_feat,)
        # in case you pass k=0 it will be incrtemented by 1
        k = cfg.k_degree[0]
        assert k != 0, "k should be > 0"
        self.k_degree = k

        
        self.eps = 1e-6
        # in the paper authors use different networks for F and G
        # self.mapper represent F
        self.mapper = Seq(Linear(in_feat, in_feat),
                          ReLU(), 
                          Linear(in_feat, in_feat)
                        )
    
        self.temperature = torch.nn.Parameter(torch.Tensor([0.001]))
        
    def forward(self, x, batch=None):
        # Case when k=0, hence only self-loop

        # Map x into some new space 
        x_g = self.mapper(x)
        # Sample edges
        edge_index, self.probs, self.mask = self.sample_edges(x_g)
        self.edge_index = edge_index
        
        return super().forward(x, edge_index)

    def sample_edges(self, nodes_emb):
        # Probabilities for edges
        probs = torch.exp(-self.temperature * torch.sum(nodes_emb - nodes_emb.unsqueeze(1), dim=-1) ** 2) + self.eps

        # Sample and extract edge indecies
        edge_index, mask = self.k_degree_sample(probs)
        
        return edge_index, probs, mask


    def k_degree_sample(self, probs):
        # prepare uniform matrix
        q = torch.rand_like(probs) 
        # Paper sampling procedure (Gumble top k)
        # Sort probabilities accross rows
        sorted_args = torch.argsort(torch.log(probs + self.eps) - torch.log(-torch.log(q + self.eps)), dim=0)

        # Take top k probs from each row
        child_nodes = sorted_args[-self.k_degree:].flatten()

        # Generate parent nodes indexes
        # min(probs.shape[0], k_degree)
        # probs.shape[0] - number of nodes, k_degree - desired degree 
        parent_nodes = torch.arange(probs.shape[0]).repeat(min(probs.shape[0], self.k_degree)).to(child_nodes.device)
        
        # Obtain Adj. mask
        mask = torch.zeros_like(probs)
        mask[child_nodes, parent_nodes] = 1
        mask[parent_nodes, child_nodes] = 1
        return torch.stack([child_nodes, parent_nodes]), mask


