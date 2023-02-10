import torch
from torch import nn
from torch.nn import functional as F
from omegaconf import DictConfig
import numpy as np

from .networks.EGnet import EGnet
from .networks.NNnet import NNnet

from models.edge_generation.EGmodule import EdgeGenerationModule

from utils import ModelTraverse 
from metrics.metrics import MetricCalculator


class Network(nn.Module):
    """Network"""
    def __init__(self, cfg: DictConfig, **args):
        """GETS cfg"""
        super().__init__()
        self.full_cfg = cfg
        self.cfg, self.args = cfg.model, args #"passes cfg.model"
        self.cfgDataloader = cfg.dataloader 
        self.cfgTrainer = cfg.trainer
        networks = {'EGnet':EGnet,
                    'NNnet': NNnet}
 
        # Initialize model
        self.model = networks.get(self.cfg.type)(self.full_cfg) # Full cfg
    
       
        # Initialize losses
        self.define_losses()
    
        # Metrics
        self.MetricCalculator = MetricCalculator(num_classes=self.cfg.outsize, device=f"cuda:{self.cfgTrainer.cuda_number}")
    
        self.init_emb()
        self.init_proto()
        
    
    def init_emb(self,):
        
        self.CatEmb = [torch.nn.Embedding(dim+1, self.cfgDataloader.imputation.cat_emb_dim).to(device=f"cuda:{self.cfgTrainer.cuda_number}")\
                    for dim in self.cfgDataloader.imputation.cat_dims]
        demoninator = len(self.cfgDataloader.imputation.cat_idx) + len(self.cfgDataloader.imputation.num_idx)
        self.num_reg = len(self.cfgDataloader.imputation.num_idx) / demoninator
        
            
        if self.cfgDataloader.imputation.cat_idx:
            self.cat_reg = len(self.cfgDataloader.imputation.cat_idx) / demoninator
        self.len_CatEmb = len(self.CatEmb)
        

    def init_proto(self,):
        if self.cfg.prototypes.k > 0:
            # To initialise prototypes it is necessary to know the exact number of dimentions after categorical values are trasformed
            self.ProtoEmbed = torch.nn.Embedding(self.cfg.prototypes.k, self.cfg.insize)
            # For each prototype initialize also its unique index, to get it from nn.Embeddings
            self.ProtoEmbedIdx = torch.Tensor(np.arange(self.cfg.prototypes.k))
            
    def define_losses(self):
       
        
        # Classification loss
        self.crossentropy = torch.nn.CrossEntropyLoss() # torch.nn.NLLLoss() 
        self.mse = torch.nn.MSELoss()

      

    def forward(self, batch):
        logits = self.model(batch)
        return logits

    def predict(self, x, y):
        # Initial number of values
        self.n_ = x.shape[0]
        y = y.squeeze(0)

        # Imputation mode
        
            
        transformedCatData = []
        for col, dense_mapper in zip(self.cfgDataloader.imputation.cat_idx, self.CatEmb):
            cat_col = x[:, col].clone().detach().type(torch.LongTensor).to(x.device)
            transformedCatData.append(dense_mapper(cat_col))

        # Workout the case when there is not categorical values
        if self.len_CatEmb > 0:
            cat_data = torch.cat(transformedCatData, dim=1)
            # Concatenate num_columns with cat columns 
            x = torch.cat([x[:, self.cfgDataloader.imputation.num_idx].clone(),
                                cat_data], dim=1)

        # Now when x is transformed it is possible to add prototypes
        if self.cfg.prototypes.k > 0:
            prototypes = self.ProtoEmbed(self.ProtoEmbedIdx.type_as(y))
            x = torch.cat([x, prototypes], dim=0)
            y = torch.cat([y, self.ProtoEmbedIdx.type_as(y)], dim=0)
            pass

        # If prototypes are trainable then make labels participate in loss: size = n_ + self.unique.size()
        logits, num_rec, cat_outputs = self.forward(x)
        
        # If prototypes were initialized it is necessary to get rid of them
        if self.cfg.prototypes.k > 0:
            logits, y, num_rec = logits[:self.n_], y[:self.n_], num_rec[:self.n_]
            cat_outputs = [cat[:self.n_] for cat in cat_outputs]
            


        assert logits.shape[0] == y.shape[0]
        return logits, y, num_rec, cat_outputs

    
    def loss_function(self, batch, mode='train'):
        """Loss fuction

        This function implements all logic during train step. In this way you
        model class is selfc ontained, hence there isn't need to change code
        in method.py when model is substituted with other one.
        """
        
            
        # Original pass                 
        logits, y, num_rec, cat_outputs = self.predict(x=batch.x, y=batch.y)
        
        m = nn.LogSoftmax(dim=1)
        loss_CE = self.crossentropy(m(logits), y)
        # batch.x_clean has all features (without any corruption) 
        loss_num, loss_cat = self.calculate_imputation_loss_wrap(x_clean=batch.x_clean,
                                                                    cat_idx=batch.cat_idx,
                                                                    num_idx=batch.num_idx,
                                                                    num_rec=num_rec, 
                                                                    cat_outputs=cat_outputs, 
                                                                    mask=batch.mask, 
                                                                    grad=True)
        
        
        loss = loss_CE + loss_cat + loss_num
        self.res_dict = {
            "loss": loss,
            "loss_cat":loss_cat,
            "loss_num": loss_num,
            "cross_entrop":loss_CE,
           
            }
        metrics_dict = self.MetricCalculator.calculate_metrics(logits=logits.detach(), labels=y.detach())
        

        ### COMPUTE ALL REGULARIZATIONS AND ADDITIONAL LOSS TERM BELOW ###
        self.special_reg(batch)

        #################################################################
        

        self.res_dict = cat_dicts(self.res_dict, metrics_dict)
        if mode != 'train':
            return self.res_dict, torch.argmax(logits.detach(), dim=1), F.softmax(logits.detach(), dim=1), num_rec, cat_outputs
        return self.res_dict, torch.argmax(logits.detach(), dim=1), F.softmax(logits.detach(), dim=1)
    
    def inference(self, x, y, batch):
        self.model.eval()
        dev = x.device
        logits, y, num_rec, cat_outputs = self.predict(x=x, y=y)
        rows, cols = batch.x.shape
        
        
        rec_x = torch.zeros((rows, cols)).to(dev)
        assert len(cat_outputs) ==  len(batch.cat_idx)
        for idx in range(len(cat_outputs)):
            col = torch.argmax(cat_outputs[idx], dim=1)
            rec_x[:, batch.cat_idx[idx]] = col

        # Writedown the preds
        rec_x[:, batch.num_idx] = num_rec

        # metrics_e2e = self.MetricCalculator.calculate_metrics(logits=logits.detach(),
        #                                                                labels=y.detach(),
        #                                                                sufix='end2end')
        probs = F.softmax(logits.detach(), dim=1)
        preds = torch.argmax(probs, dim=1)
        return probs, preds, y, rec_x


    def special_reg_func(self, module, **args):
        A =((args['batch'].y.unsqueeze(0) == args['batch'].y.unsqueeze(1)) * 1)
        reg = self.cfg.prob_reg * (module.StatDict['dist_mat'] * A).mean()
        self.res_dict['reg'] = reg
        self.res_dict['loss'] += reg

    def SupervisedReg_func(self, module, **args):
        A =((args['batch'].y.unsqueeze(0) == args['batch'].y.unsqueeze(1)) * 1)
        mask = module.StatDict['mask']   
        reg = self.cfg.prob_reg * (A - (A*mask)).mean()
        self.res_dict['reg'] = reg
        self.res_dict['loss'] += reg     

    def SupervisedReg_onlywrong_func(self, module, **args):
        A =((args['batch'].y.unsqueeze(0) == args['batch'].y.unsqueeze(1)) * 1)
        A_hat = (1-A)
        mask = module.StatDict['mask']    
        
        # Case with prototypes
        if mask.shape[0] > A_hat.shape[0]:
            cur_n = A_hat.shape[0]
            mask = mask[:cur_n,:cur_n]

        reg = self.cfg.prob_reg * (A_hat * mask).mean()
        self.res_dict['reg'] = reg
        self.res_dict['loss'] += reg 
    
    def special_reg(self, batch):
        if self.cfg.reg_type == 'paper':
            ModelTraverse(model=self.model, 
                        SearchModule=EdgeGenerationModule,
                        func=self.special_reg_func, batch=batch
                        )
        if self.cfg.reg_type == 'A-mask':
            ModelTraverse(model=self.model, 
                        SearchModule=EdgeGenerationModule,
                        func=self.SupervisedReg_func, batch=batch
                        )
        if self.cfg.reg_type == 'A_hat*mask':
            ModelTraverse(model=self.model, 
                        SearchModule=EdgeGenerationModule,
                        func=self.SupervisedReg_onlywrong_func, batch=batch
                        )

##############################################################################    
    def calculate_imputation_loss_wrap(self, x_clean, num_rec, cat_outputs, mask, cat_idx, num_idx, grad=True):
        if not grad:
            with torch.no_grad():
                output = self.calculate_imputation_loss(x_clean, num_rec, cat_outputs, mask, cat_idx, num_idx)
        else:
            output = self.calculate_imputation_loss(x_clean, num_rec, cat_outputs, mask, cat_idx, num_idx)
        loss_num, loss_cat = output
        return loss_num, loss_cat
            

    def calculate_imputation_loss(self, x_clean, num_rec, cat_outputs, mask, cat_idx, num_idx):
        # Numerical loss
        if len(num_idx) > 0:
            mask_num = (1 - mask[:, num_idx])
            # Numerical loss
            num_corrupted_idxs = mask_num==1
            x_num = x_clean[:, num_idx][num_corrupted_idxs] # Such indexing gives vectors with size: (n_corrupted_values, 1)
            x_num_rec = num_rec[num_corrupted_idxs]
            if len(x_num_rec) == 0:
                loss_num = torch.tensor([0]).type(x_num.type())
            else:
                loss_num = self.num_reg * self.mse(x_num, x_num_rec)
        else: loss_num = torch.tensor([0]).type(x_clean.type())

        # Categorical loss
        if len(cat_idx) > 0:
            mask_cat = (1 - mask[:, cat_idx])
            loss_cat = []
            x_cat = x_clean[:, cat_idx].type(torch.LongTensor).to(x_clean.device)
            for out, col in zip(cat_outputs, np.arange(mask_cat.shape[1])):
                corrupted_idxs = (mask_cat[:, col]==1)
                if corrupted_idxs.sum() > 0:
                    x_cat_rec = out[corrupted_idxs]
                    
                    loss_cat.append(self.crossentropy(x_cat_rec, x_cat[corrupted_idxs, col]))
            # Normalize with respect to number categorical columns which have been processed
            if len(loss_cat)==0:
                loss_cat = torch.tensor([0]).type(x_clean.type())
            else:
                loss_cat = self.cat_reg * sum(loss_cat) / len(loss_cat)
        else: loss_cat = torch.tensor([0]).type(x_clean.type())
        return loss_num, loss_cat



def cat_dicts(a, b):
    return dict(list(a.items()) + list(b.items()))

def kl_categorical(preds, log_prior, num_atoms, eps=1e-16):
    kl_div = preds * (torch.log(preds + eps) - log_prior)
    return kl_div.sum() / (num_atoms * preds.size(0))


def kl_categorical_uniform(preds, num_atoms, num_edge_types=2, add_const=False,
                           eps=1e-16):
    kl_div = preds * torch.log(preds + eps)
    if add_const:
        const = np.log(num_edge_types)
        kl_div += const
    return kl_div.sum() / (num_atoms * 1)

def init_proto(self,):
    if self.cfg.prototypes.k > 0:
        # To initialise prototypes it is necessary to know the exact number of dimentions after categorical values are trasformed
        self.ProtoEmbed = torch.nn.Embedding(self.cfg.prototypes.k, self.cfg.insize)
        # For each prototype initialize also its unique index, to get it from nn.Embeddings
        self.ProtoEmbedIdx = torch.Tensor(np.arange(self.cfg.prototypes.k))

def predict(self, x, y):
    # Initial number of values
    self.n_ = x.shape[0]
    y = y.squeeze(0)

    # Imputation mode
    
    transformedCatData = []
    for col, dense_mapper in zip(self.cfgDataloader.imputation.cat_idx, self.CatEmb):
        cat_col = x[:, col].clone().detach().type(torch.LongTensor).to(x.device)
        transformedCatData.append(dense_mapper(cat_col))

    # Workout the case when there is not categorical values
    if self.len_CatEmb > 0:
        cat_data = torch.cat(transformedCatData, dim=1)
        # Concatenate num_columns with cat columns 
        x = torch.cat([x[:, self.cfgDataloader.imputation.num_idx].clone(),
                            cat_data], dim=1)

    # Now when x is transformed it is possible to add prototypes
    if self.cfg.prototypes.k > 0:
        prototypes = self.ProtoEmbed(self.ProtoEmbedIdx.type_as(y))
        x = torch.cat([x, prototypes], dim=0)
        y = torch.cat([y, self.unique.type_as(y)], dim=0)
        pass

    # If prototypes are trainable then make labels participate in loss: size = n_ + self.unique.size()
    logits, num_rec, cat_outputs = self.forward(x)
    
    # If prototypes were initialized it is necessary to get rid of them
    if self.cfg.prototypes.k > 0:
        logits, y = logits[:self.n_], y[:self.n_]
        num_rec, cat_outputs = num_rec[:self.n_], cat_outputs[:self.n_]


    assert logits.shape[0] == y.shape[0]
    return logits, y, num_rec, cat_outputs







# def cat_dicts(a, b):
#     return dict(list(a.items()) + list(b.items()))

# class Network(nn.Module):
#     """Network"""
#     def __init__(self, cfg: DictConfig, **args):
#         """GETS cfg"""
#         super().__init__()
#         self.args = args
#         self.full_cfg = cfg
#         self.model_params = cfg.model
#         self.dataset_params = cfg.dataloader  
        
#         self.device = f"cuda:{cfg.trainer.cuda_number}" 
        
#         networks = { 
#             'EGnet':EGnet,
#             'NNnet': NNnet
#             }
 
#         # Initialize model
#         self.model = networks.get(self.model_params.type)(self.full_cfg) # Full cfg    

#         # Initialize losses
#         self.define_losses()
    
#         # Metrics
#         self.MetricCalculator = MetricCalculator(num_classes=self.model_params.outsize, device=self.device)
        
#         self.init_emb()
#         self.init_proto()
        
    
#     def init_emb(self,):
#         self.CatEmb = [torch.nn.Embedding(dim + 1, self.dataset_params.imputation.cat_emb_dim).to(device=self.device)\
#                     for dim in self.dataset_params.imputation.cat_dims]
        
#         demoninator = len(self.dataset_params.imputation.cat_idx) + len(self.dataset_params.imputation.num_idx) 
#         self.num_reg = len(self.dataset_params.imputation.num_idx) / demoninator
        
#         if self.dataset_params.imputation.cat_idx:
#             self.cat_reg = len(self.dataset_params.imputation.cat_idx) / demoninator
#         else:
#             self.cat_reg = 0
#         self.len_CatEmb = len(self.CatEmb)
        
#     def init_proto(self,):
#         if self.model_params.prototypes.k > 0:
#             # To initialise prototypes it is necessary to know the exact number of dimentions after categorical values are trasformed
#             self.ProtoEmbed = torch.nn.Embedding(self.model_params.prototypes.k, self.model_params.insize)
#             # For each prototype initialize also its unique index, to get it from nn.Embeddings
#             self.ProtoEmbedIdx = torch.Tensor(np.arange(self.model_params.prototypes.k))
            
#     def define_losses(self):        
#         # Classification loss
#         self.crossentropy = torch.nn.CrossEntropyLoss()
#         self.mse = torch.nn.MSELoss()

#     def loss_function(self, batch, mode='train'):
#         """Loss fuction

#         This function implements all logic during train step. In this way you
#         model class is selfcontained, hence there isn't need to change code
#         in method.py when model is substituted with other one.
#         """
        
#         # Original pass       
#         y = batch.y
#         logits, num_rec, cat_outputs = self.predict(x=batch.x)
#         loss_CE = self.crossentropy(logits, y)
        
#         # mask - during training and validation mask_arti
#         # during testing mask_init is not used
#         loss_num, loss_cat = self.calculate_imputation_loss(x_clean=batch.x_clean,
#                                                             cat_idx=batch.cat_idx,
#                                                             num_idx=batch.num_idx,
#                                                             num_rec=num_rec, 
#                                                             cat_outputs=cat_outputs, 
#                                                             mask=batch.mask)

#         loss = loss_CE + (self.cat_reg * loss_cat) + (self.num_reg * loss_num)

#         # Loss
#         self.res_dict = {
#             "loss": loss,
#             "loss_cat": loss_cat,
#             "loss_num": loss_num,
#             "cross_entrop": loss_CE,
#             }
#         # Regularization loss
#         self.homo_reg(batch)
        
#         probs= F.softmax(logits.detach(), dim=-1)
#         preds = torch.argmax(logits.detach(), dim=-1)
#         # Metrics
#         metrics_dict = self.MetricCalculator.calculate_metrics(probs=probs.detach(),
#                         preds=preds.detach(),
#                         labels=y.detach()
#                         )
            
        
#         self.res_dict = cat_dicts(self.res_dict, metrics_dict)

#         if mode in ['test', 'val']:
#             return self.res_dict, preds, probs, num_rec, cat_outputs

#         return self.res_dict, preds, probs 

#     def predict(self, x):
#         # Initial number of values
#         self.n_ = x.shape[0]
    
#         # Imputation mode    
#         transformedCatData = []
#         for col, dense_mapper in zip(self.dataset_params.imputation.cat_idx, self.CatEmb):
#             cat_col = x[:, col].clone().detach().type(torch.LongTensor).to(x.device)
#             transformedCatData.append(dense_mapper(cat_col))

#         # Workout the case when there is not categorical values
#         if self.len_CatEmb > 0:
#             cat_data = torch.cat(transformedCatData, dim=1)
#             # Concatenate num_columns with cat columns 
#             x = torch.cat([x[:, self.dataset_params.imputation.num_idx].clone(),
#                                 cat_data], dim=1)

#         # Now when x is transformed it is possible to add prototypes
#         if self.model_params.prototypes.k > 0:
#             prototypes = self.ProtoEmbed(self.ProtoEmbedIdx.type(torch.LongTensor).to(x.device))
#             x = torch.cat([x, prototypes], dim=0)
            
    
#         logits, num_rec, cat_outputs = self.model(x)
        
#         # Prototypes contribute through the neighbourhood agg. 
#         # Get rid of prototypes from logits and reconstructuions.
#         if self.model_params.prototypes.k > 0:
#             logits, num_rec = logits[:self.n_], num_rec[:self.n_]
#             cat_outputs = [cat[:self.n_] for cat in cat_outputs]
#             assert logits.shape[0] == self.n_
            
#         return logits, num_rec, cat_outputs
    
#     def homo_reg(self, batch):
#         if self.model_params.reg_type == 'A_hat*mask':
#             ModelTraverse(model=self.model, 
#                         SearchModule=EdgeGenerationModule,
#                         func=self.adj_reg, batch=batch
#                         )

#     def adj_reg(self, module, **args):
#         # Create homogeneous edges
#         A =((args['batch'].y.unsqueeze(0) == args['batch'].y.unsqueeze(1)) * 1)
#         A_hat = (1-A)
#         # True mask 
#         mask = module.StatDict['mask']    
        
#         # Case with prototypes
#         if mask.shape[0] > A_hat.shape[0]:
#             cur_n = A_hat.shape[0]
#             # Do not consider prototypes in the loss 
#             mask = mask[:cur_n,:cur_n]

#         reg = self.model_params.prob_reg * (A_hat * mask).mean()
        
#         # Collect statistics
#         self.res_dict['reg'] = reg
#         self.res_dict['loss'] += reg 
            

#     def calculate_imputation_loss(self, x_clean, num_rec, cat_outputs, mask, cat_idx, num_idx):
#         # Numerical loss
#         if len(num_idx) > 0:
            
#             mask_num = mask[:, num_idx]

#             # Numerical loss
#             # mask := {0-corruped, 1-clean}
#             num_corrupted_idxs = (mask_num==0)
#             x_num = x_clean[:, num_idx][num_corrupted_idxs] # Such indexing gives vectors with size: (n_corrupted_values, 1)
#             x_num_rec = num_rec[num_corrupted_idxs]

#             if len(x_num_rec) == 0:
#                 loss_num = torch.tensor(0).type(x_num.type())
#             else:
#                 loss_num = self.mse(x_num, x_num_rec)
            
#         else: loss_num = torch.tensor([0]).type(x_clean.type())

#         # Categorical loss
#         if len(cat_idx) > 0:
#             mask_cat = mask[:, cat_idx]
#             loss_cat = []
#             x_cat = x_clean[:, cat_idx].type(torch.LongTensor).to(x_clean.device)
#             for out, col in zip(cat_outputs, np.arange(mask_cat.shape[1])):
#                 corrupted_idxs = (mask_cat[:, col]==0)
#                 if corrupted_idxs.sum() > 0:
#                     x_cat_rec = out[corrupted_idxs]
#                     loss_cat.append(self.crossentropy(x_cat_rec, x_cat[corrupted_idxs, col]))
            
#             # Normalize with respect to number categorical columns which have been processed
#             if len(loss_cat) > 0:
#                 loss_cat = sum(loss_cat) / len(loss_cat)
#             else: loss_cat = torch.tensor(0).type(x_clean.type())

#         else: loss_cat = torch.tensor([0]).type(x_clean.type())

#         return loss_num, loss_cat



