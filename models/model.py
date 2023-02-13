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

def cat_dicts(a, b):
    return dict(list(a.items()) + list(b.items()))

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
        self.crossentropy = torch.nn.CrossEntropyLoss()
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

        output = self.calculate_imputation_loss(x_clean=batch.x_clean, 
                                                num_rec=num_rec, 
                                                cat_outputs=cat_outputs,
                                                mask=batch.mask,
                                                cat_idx=batch.cat_idx,
                                                num_idx=batch.num_idx)
        loss_num, loss_cat = output
        
        
        loss = loss_CE + loss_cat + loss_num
        self.res_dict = {
            "loss": loss,
            "loss_cat":loss_cat,
            "loss_num": loss_num,
            "cross_entrop":loss_CE,
           
            }
        metrics_dict = self.MetricCalculator.calculate_metrics(logits=logits.detach(), labels=y.detach())
        

        ### Cimpute reg. for every layer###
        self.special_reg(batch)


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

        
        probs = F.softmax(logits.detach(), dim=1)
        preds = torch.argmax(probs, dim=1)
        return probs, preds, y, rec_x

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
        if self.cfg.reg_type == 'A_hat*mask':
            ModelTraverse(model=self.model, 
                        SearchModule=EdgeGenerationModule,
                        func=self.SupervisedReg_onlywrong_func, batch=batch
                        )


            

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




