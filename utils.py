import re
import numpy as np
import torch
from omegaconf import DictConfig

from dataloader import PL_DataModule

def get_dataloader(cfg: DictConfig, data=None):
    return PL_DataModule(cfg.dataloader, model_type=cfg.model.edge_generation_type, data=data)

# -------------------- #             
def ModelTraverse(model, SearchModule, func, **args):
        for NameModule in model.named_modules():
                name, module = NameModule
                if isinstance(module, SearchModule):  
                        args['module_number'] = ExtractModuleNumber(name)  
                        func(module, **args)

def ExtractModuleNumber(name):
    m = re.search('EGG_(.+?).', name)
    if not m:
        return None
        
    return m.group(1)

def TauSetup(module, **args):
    module.cfg = args['cfg']                       

def TauUpdate(module, **args):
    module.current_step = args['global_step'] 
        
def init_weights(module, **args):
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.xavier_uniform(module.weight)
        module.bias.data.fill_(0.01)

def torch2numpy(x):
    return x.detach().cpu().numpy()


def config_preprocess(cfg, datamodule):
    # Obtain feature sizes and number of labels
    batch = next(iter(datamodule.train_dataloader()))
    cfg.model.opt.loader_batches = len(datamodule.train_dataloader())
    cfg.model.insize ,cfg.model.outsize = batch.x.shape[-1], len(np.unique(datamodule.train_dataloader().dataset.labels))
    cfg.dataloader.imputation.num_idx = [i.item() for i in batch.num_idx.numpy()]
    cfg.dataloader.imputation.cat_idx = [i.item() for i in batch.cat_idx.numpy()]
    cfg.dataloader.imputation.cat_dims = [i.item() for i in batch.cat_dims.numpy()]
    cfg.dataloader.imputation.CatFillWith = [int(i.item()) for i in batch.CatFillWith.numpy()] 
    cfg.dataloader.imputation.MostFreqClass = [int(i.item()) for i in batch.MostFreqClass.numpy()] 
    cfg.model.insize = len(cfg.dataloader.imputation.num_idx)\
                    + len(cfg.dataloader.imputation.cat_idx) * cfg.dataloader.imputation.cat_emb_dim
    return cfg 