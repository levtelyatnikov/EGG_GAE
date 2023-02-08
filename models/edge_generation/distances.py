import torch

class DistanceBlock(torch.nn.Module):
    def __init__(self, **args):
        super(DistanceBlock, self).__init__()
        self.eps = 10e-10

    def forward(self, x1, x2, temperature):
        "Implement"
        pass

class L2_dist(DistanceBlock):
    def __init__(self, **args):
        super(L2_dist, self).__init__()
        self.temperature = torch.nn.Parameter(torch.Tensor([0.1]))
        self.eps = 1e-6
    def forward(self, x1, x2,):
        #normalized_x1 = torch.nn.functional.normalize(x1)  
        #normalized_x2 = torch.nn.functional.normalize(x2)  
        dist_mat = torch.sum((x1 - x2.unsqueeze(1)) ** 2 , dim=-1) + self.eps
        probs = torch.exp(-self.temperature * dist_mat) 
        return probs, dist_mat 

class cos_dist(DistanceBlock):
    def __init__(self, **args):
        super(cos_dist, self).__init__()
        
    def forward(self, x1, x2, temperature):
        dist_mat = ((self.pw_cosine_distance(x1, x2) + self.eps) ** 2)
        probs = torch.exp(-temperature * dist_mat)
        return probs, dist_mat 
    
    def pw_cosine_distance(self, x1, x2):
        normalized_x1 = torch.nn.functional.normalize(x1)  
        normalized_x2 = torch.nn.functional.normalize(x2)  
        res = (1 - torch.mm(normalized_x1, normalized_x2.T)) + self.eps
        return res
