import torch
from ..edge_generation.distances import *

class EdgeSampling(torch.nn.Module):
    def __init__(self, **args):
        super(EdgeSampling, self).__init__()
        GumbleDistFunc = {
                        "L2_dist": L2_dist,}
        
        self.cfg = args['cfg'] # """Gets cfg.model"""
        self.dist = GumbleDistFunc.get(args['GumbleDistFunc_type'])(**args) 
        self.eps = 1e-6

    def forward(self, x, tau, temperature):
        #self.tau = tau
        #self.temperature = temperature
        return self.sample_edges(nodes_emb=x, tau=tau, hard=True)

    def calculate_prob_dist(self, x1, x2):
        probs, dist_mat = self.dist(x1=x1, x2=x2) 

        return probs, dist_mat

    def sample_edges(self, nodes_emb, **args):
        """Implement"""
        pass

class GumbelSigmoid(EdgeSampling):  
    def __init__(self, **args):
        super(GumbelSigmoid, self).__init__(**args)
    
    def sample_edges(self, nodes_emb, **args): # nodes_emb, tau, hard=True
        tau = args['tau']
        # CHANGE INTO SYM AND NON SYM MODULES
        # =====Torch version=======
        probs, dist_mat = self.calculate_prob_dist(x1=nodes_emb, x2=nodes_emb)
        probs = torch.triu(probs, diagonal=1)
        probs = torch.cat([probs.unsqueeze(-1), (1 - probs).unsqueeze(-1)], dim=-1)

        logits = torch.log(probs + self.eps)

        A = torch.nn.functional.gumbel_softmax(logits + self.eps, tau=tau, hard=True, dim= -1)[:,:, 0]
    
        A = (torch.triu(A, diagonal=1) + torch.triu(A, diagonal=1).T) #A + A.T
        a_shape = A.shape[0]
        diag = torch.eye(a_shape).type_as(A)
        ones = torch.ones(a_shape).type_as(A)
        A = A * (ones - diag) + diag

        # Get parent child list
        childer = torch.arange(a_shape).repeat(a_shape).to(nodes_emb.device).detach()
        parents = torch.arange(a_shape).view(-1,1).repeat((1, a_shape)).flatten().to(nodes_emb.device).detach()
        edge_index = torch.stack([childer, parents])

        # Get weight sequence of 0, 1 for edge_list
        probs = probs[:, :, 0]
        assert A.max()==1, f"current P.max() = {A.max()}"

        edge_weights = A.view((-1, ))

        return edge_index, edge_weights, probs, A, dist_mat

class GumbelSigmoid_k(EdgeSampling):
    def __init__(self, **args):
        super(GumbelSigmoid_k, self).__init__(**args)
        self.k_degree = args['k_degree']
        self.temperature = torch.nn.Parameter(torch.Tensor([0.001]))

    def sample_edges(self, nodes_emb, **args): 
        # Probabilities for edges
        dist_mat = torch.sum(nodes_emb - nodes_emb.unsqueeze(1), dim=-1) ** 2
        probs = torch.exp(-self.temperature * dist_mat) + self.eps

        # Sample and extract edge indecies
        edge_index, mask = self.k_degree_sample(probs)
        
        return edge_index, torch.ones(edge_index.shape[1]).to(edge_index.device), probs, mask, dist_mat

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
    