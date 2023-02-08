import torch
import torch_cluster
from torch_geometric.nn import MessagePassing
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.typing import OptTensor

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

# --------------------DynamicEdgeConv---------- ----------  
class DynamicEdgeConv(EdgeConv):
    def __init__(self, in_channels, out_channels, **args):
        super().__init__(in_channels, out_channels, **args)
        self.k_degree = args['k_degree']
        
    def forward(self, x, batch=None):
        # Case when k = 1, hence only self-loop
        if self.k_degree == "selfloop":
            # return self correspondence [[0, 1, 2,], [0, 1, 2]]
            edge_index = torch_cluster.knn_graph(x, 1, batch=None,
                                                 loop=True,
                                                 flow='source_to_target') 
        elif self.k_degree >= 1:
            edge_index = torch_cluster.knn_graph(x, self.k_degree,
                                                 batch=None, loop=False,
                                                 flow='source_to_target')
        else: print('Ups...')
        self.generate_prob_mask(edge_index, x)
        self.edge_index = edge_index
        return super().forward(x, edge_index)
        
    def generate_prob_mask(self, edge_index, x):
        self.StatDict = {}
        max_val = x.shape[0] 
        M = torch.zeros((max_val, max_val)).type_as(edge_index).float()
        M[edge_index[0], edge_index[1]] = torch.tensor(1).type_as(edge_index).float()
        self.StatDict['mask'] = M
        self.StatDict['probs'] = M



    
# --------------------DynamicEdgeConv_DGM--------------------
class DynamicEdgeConv_DGM(EdgeConv):
    def __init__(self, in_feat, out_feat, cfg, **args):
        super().__init__(in_feat, out_feat)


        # in case you pass k=0 it will be incrtemented by 1
        k = cfg.k_degree[0]
        if k == 0:
            self.k_degree = k + 1
            print("Avoid using DynamicEdgeConv_DGM with k = 0")
            print(f"k is incremented by one, hence k = {self.k_degree}")
        else: 
            self.k_degree = k

        self.eps = 1e-6
        
        
        self.mapper = Seq(Linear(in_feat, in_feat),
                          ReLU(), 
                          Linear(in_feat, in_feat)
                        )
        # initialize temp. parameter with 1, however not sure this is a good idea :)
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


# kEGG-GAE
class kEGG_GAE(EdgeConv):
    def __init__(self, in_channels, out_channels, cfg, **args):
        super().__init__(in_channels, out_channels)
        # in case you pass k=0 it will be incrtemented by 1
        k = cfg.k_degree[0]
        if k == 0:
            self.k_degree = k + 1
            print("Avoid using DynamicEdgeConv_DGM with k = 0")
            print(f"k is incremented by one, hence k = {self.k_degree}")
        else: 
            self.k_degree = k

        
        self.eps = 1e-6
        # in the paper authors use different networks for F and G
        # self.mapper represent F
        self.mapper = Seq(Linear(in_channels, in_channels),
                          ReLU(), 
                          Linear(in_channels, in_channels)
                        )
        # initialize temp. parameter with 1, however not sure this is a good idea :)
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

# # --------------------DynamicEdgeConv_MINE--------------------
# class EdgeConvMINE(MessagePassing):
#     def __init__(self, in_channels, out_channels, k):
#         super().__init__(aggr='mean') #  "Max" aggregation.
#         self.mlp = Seq(Linear(2 * in_channels, out_channels),
#                        ReLU(),
#                        Linear(out_channels, out_channels))
#         # if k == 0
#         # self.mlp_k0 = Seq(Linear(in_channels, out_channels),
#         #                ReLU(),
#         #                Linear(out_channels, out_channels))

#     def forward(self, x, edge_index):
#         # x has shape [N, in_channels]
#         # edge_index has shape [2, E]
       
#         return self.propagate(edge_index, x=x)

#     def message(self, x_i, x_j):
#         # x_i has shape [E, in_channels]
#         # x_j has shape [E, in_channels] 
#         tmp = torch.cat([x_i, x_j], dim=1)  # tmp has shape [E, 2 * in_channels]
#         return self.mlp(tmp)

# class DynamicEdgeConvKNN(EdgeConvMINE):
#     def __init__(self, in_channels, out_channels, k):
#         super().__init__(in_channels, out_channels, k)
#         self.k_degree = k
#         self.mapper = Seq(Linear(in_channels, in_channels),
#                           ReLU(), 
#                           Linear(in_channels, in_channels)
#                         )
        
#     def forward(self, x, batch=None):
#         x_g = self.mapper(x)
#         # Case when k = 1, hence only self-loop
#         if self.k_degree == 'selfloop':
#             # return self correspondence [[0, 1, 2,], [0, 1, 2]]
#             edge_index = torch_cluster.knn_graph(x_g, 1, batch=None,
#                                                  loop=True,
#                                                  flow='source_to_target') 
#         elif self.k_degree >= 1:
#             edge_index = torch_cluster.knn_graph(x_g, self.k_degree, batch=None,
#                                                  loop=False,
#                                                  flow='source_to_target')
#         else: print('Ups...')
    
#         self.edge_index = edge_index
#         return super().forward(x_g, edge_index)