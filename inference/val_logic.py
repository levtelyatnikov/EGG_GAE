import torch
from pytorch_lightning import Callback
from collections import defaultdict

from inference.ensembler import Ensemble
from inference.utils import cat_dicts

import numpy as np

def rmse_f(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

class ValCallback(Callback):
    def __init__(self, num_classes, device):
        super().__init__()
        self.agregator = Ensemble(num_classes=num_classes, device=device)

    def create_collector(self,):
        self.pred_collector = {'preds':defaultdict(list),
                               'probs':defaultdict(list), 
                               'mapper': defaultdict(int)}
        
        self.recon_collector = {'num': defaultdict(list),
                               'cat': defaultdict(list), 
                               'mapper': defaultdict(int)}
        self.batch_val_loss_dicts = [] 
    
    def on_validation_epoch_start(self, trainer, pl_module):
        # Create cillector at the beggining of the validation epoch
        self.create_collector()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        val_loss, preds, probs, num_rec, cat_outputs = outputs 

        # After each epoch statistics of the batch inserted into stat_collector
        # Update stat collector
        self.updata_stat(batch.unique_idx, batch.y, preds, probs)
        self.update_recon_collector(batch, num_rec, cat_outputs)
        
        # Collect val_loss dict, which consist of 
        self.batch_val_loss_dicts.append(val_loss)
        return val_loss

    def log_validation_epoch_end(self, trainer):
        logs = {}
        
        keys = self.batch_val_loss_dicts[0].keys()
        for key in keys:
            logs["val_" + key] = torch.stack([x[key] for x in self.batch_val_loss_dicts]).mean()
        
        return logs

    def on_validation_epoch_end(self, trainer, pl_module):
        ## Log vall loss, how to avoid this mess?
        val_logs = self.log_validation_epoch_end(trainer=trainer)
        
        ## Log validation and test for ensemble and vouting
        log_dict = {}

        min_mean_predictions = pl_module.datamodule.cfg.min_sample_predictions
        pred_ens_collector, imp_ens_collector = self.agregator.proscess(pred_collector=self.pred_collector,
                 k=min_mean_predictions,
                 imp_collector=self.recon_collector
                 )
                 
        assert pred_ens_collector["min_predictions"]==min_mean_predictions
        val_dataset = pl_module.datamodule.val_dataloader().dataset
        
        # Validation dataset is always with MCAR noise 
        # (the noise level is at level 0.2).
        # X_imputed, rmse, acc = self.impute_dataset(dataset=val_dataset,
        #                     MASK=val_dataset.MASK_init, # MASK_val
        #                     imp_ens_collector=imp_ens_collector,
        #                     num_idx=val_dataset.num_idx,
        #                     cat_idx=val_dataset.cat_idx)

        for key in pred_ens_collector.keys():
            log_dict["val_" + key] = pred_ens_collector[key]
        
        ## Update maximum VAL 
        log_dict = cat_dicts(val_logs, log_dict)
        self.log_dict(log_dict)
    
    def impute_dataset(self, dataset, MASK, imp_ens_collector, num_idx, cat_idx):
        X_imputed = dataset.features.copy()
        rmse, acc = 0, 0
        if len(num_idx)>0:
            data_n = dataset.features_clean[:, num_idx]
            mask_n = MASK[:, num_idx]

            row, col = (mask_n == 0).nonzero().numpy().T
            row_col = list(zip(row,col))
            row_col_pred = imp_ens_collector['num'].keys()

            assert set(row_col) == set(row_col_pred)
            target = data_n[row, col]
            preds = np.array([imp_ens_collector['num'][key] for key in row_col])
            rmse = rmse_f(preds, target)

            # Impute dataset
            data_imp = X_imputed[:, num_idx].copy()
            data_imp[row, col] = preds
            X_imputed[:, num_idx] = data_imp

        
        if len(cat_idx)>0:
            data_c = dataset.features_clean[:, cat_idx]            
            mask_c = MASK[:, cat_idx]

            row, col = (mask_c == 0).nonzero().numpy().T
            row_col = list(zip(row, col))
            row_col_pred = imp_ens_collector['cat'].keys()

            assert set(row_col) == set(row_col_pred)

            target = data_c[row, col]
            preds = np.array([imp_ens_collector['cat'][key] for key in row_col])
            acc = (target == preds).sum()/len(target)

            # Impute dataset
            data_imp = X_imputed[:, cat_idx].copy()
            data_imp[row, col] = preds
            X_imputed[:, cat_idx] = data_imp
        
        return X_imputed, rmse, acc


    def update_recon_collector(self, batch, num_rec, cat_rec):
        if len(batch.num_idx)>0:
            # batch row, column
            assert batch.mask[:, batch.num_idx].shape == num_rec.shape
            corrupt_idx = (batch.mask[:, batch.num_idx] == 0).nonzero()
            
            # True row and column indices in x_clean[:, num_idx]
            datum_idxs = batch.unique_idx[corrupt_idx[:, 0]].detach().cpu().numpy()
            column = corrupt_idx[:, 1].detach().cpu().numpy()
            keys = list(zip(datum_idxs, column))
            
            # Extract imputed values
            values = num_rec[corrupt_idx[:,0], corrupt_idx[:,1]]

    
            assert len(keys) == len(values)
            for key, val in zip(keys, values):
                self.recon_collector['num'][key].append(val)
        
        if len(batch.cat_idx)>0:
            assert len(cat_rec) == len(batch.cat_idx.cpu())
            # batch.cat_idx - consists columns of categorical features in x_clean
            # in fact it is easier to 
            mask_cat = batch.mask[:, batch.cat_idx]
            for out, col in zip(cat_rec, np.arange(len(batch.cat_idx))):
            
                #
                corrupt_idx = (mask_cat[:, col] == 0).nonzero()
                assert corrupt_idx.shape[1] == 1
                corrupt_idx = corrupt_idx.view(-1)
                # True row and column indices in x_clean[:, num_idx]
                datum_idxs = batch.unique_idx[corrupt_idx].detach().cpu().numpy()
                
                keys = [(datum, col) for datum in datum_idxs]

                
                if corrupt_idx.sum() > 0:
                    assert len(keys) == out[corrupt_idx].shape[0]
                    for key, logits in zip(keys, out[corrupt_idx]):
                        self.recon_collector['cat'][key].append(logits.detach().cpu())

    def updata_stat(self, unique_idx, labels, preds, probs):
        unique_idx, labels, preds, probs = unique_idx.view(-1,),\
                                           labels.view(-1,),\
                                           preds.view(-1,),\
                                           probs # probs has dim: batch_size, n_classes
      
        # Collect each prediction into its pool
        # according to unique_idx of the datum.
        for idx, label, pred, prob in zip(unique_idx, labels, preds, probs, ):
            idx = idx.item()
            self.pred_collector['preds'][idx].append(pred)
            self.pred_collector['probs'][idx].append(prob.unsqueeze(0))
            self.pred_collector['mapper'][idx] = label
