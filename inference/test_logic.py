import torch
from pytorch_lightning import Callback
from collections import defaultdict
import numpy as np
from utils import torch2numpy
from inference.ensembler import Ensemble
from inference.utils import cat_dicts



def rmse_f(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

class TestCallback(Callback):
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
        self.batch_test_loss_dicts = [] 
    
    def on_test_epoch_start(self, trainer, pl_module):
        # Create cillector at the beggining of the testidation epoch
        self.create_collector()

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        test_loss, preds, probs, num_rec, cat_outputs = outputs 

        # After each epoch statistics of the batch inserted into stat_collector
        # Update stat collector
        self.updata_stat(batch.unique_idx, batch.y, preds, probs)
        self.update_recon_collector(batch, num_rec, cat_outputs)
        
        # Collect test_loss dict, which consist of 
        self.batch_test_loss_dicts.append(test_loss)
        return test_loss

    def log_test_epoch_end(self, trainer):
        logs = {}
        
        keys = self.batch_test_loss_dicts[0].keys()
        for key in keys:
            logs["test_" + key] = torch.stack([x[key] for x in self.batch_test_loss_dicts]).mean()
        
        return logs

    def on_test_epoch_end(self, trainer, pl_module):
        ## Log testl loss, how to avoid this mess?
        test_logs = self.log_test_epoch_end(trainer=trainer)
        
        ## Log testidation and test for ensemble and vouting
        log_dict = {}
        
        # Get current datasplit
        mode = trainer.test_dataloaders[0].dataset.split
        if mode == 'train_test':
            mode='train'

        min_mean_predictions = pl_module.datamodule.cfg.min_sample_predictions
        pred_ens_collector, imp_ens_collector = self.agregator.proscess(pred_collector=self.pred_collector,
                 k=min_mean_predictions,
                 imp_collector=self.recon_collector
                 )
                 
        assert pred_ens_collector["min_predictions"]==min_mean_predictions
        
        # Impute dataset
        test_dataset = trainer.test_dataloaders[0].dataset 
        X_imputed, y = self.impute_dataset(dataset=test_dataset,
                            MASK=test_dataset.MASK_init,
                            imp_ens_collector=imp_ens_collector,
                            num_idx=test_dataset.num_idx,
                            cat_idx=test_dataset.cat_idx)

        for key in pred_ens_collector.keys():
            log_dict[f"{mode}_" + key] = pred_ens_collector[key]
        
        dataset = trainer.test_dataloaders[0].dataset
        
        X_clean = dataset.features_clean
        num_idx = dataset.num_idx
        cat_idx = dataset.cat_idx
        mask_init = dataset.MASK_init.numpy()
        rmse, mae, acc = compute_imp_metrics(X_clean=X_clean, 
            X_imp=X_imputed,
            mask=mask_init,
            num_idx=num_idx,
            cat_idx=cat_idx
            )
        # # Calculate same stat for statistics predictions
        

        ## Update maximum test 
        
        log_dict = cat_dicts(test_logs, log_dict)
        
        log_dict[f"{mode}-rmse"] = rmse
        log_dict[f"{mode}-mae"] = mae
        log_dict[f"{mode}-acc"] = acc

        self.log_dict(log_dict)


        if mode == 'train':
            mode_clf = 'tr'
        else:
            mode_clf = mode
            
        result_dict, data_dict, e2e_stat = {}, {}, {f'{mode_clf}':{}}


        result_dict[f"{mode}-rmse"] = rmse
        result_dict[f"{mode}-mae"] = mae
        result_dict[f"{mode}-acc"] = acc

        data_dict[f'{mode}_imp'] = X_imputed
        data_dict [f'y_{mode}'] = y
        

        # Collect e2e stat
        if mode == 'train':
            mode_clf = 'tr'
        else:
            mode_clf = mode
        e2e_stat[f'{mode_clf}']['accuracy'] = log_dict[f'{mode}_ens_accuracy'].cpu().numpy()
        e2e_stat[f'{mode_clf}']['f1'] = log_dict[f'{mode}_ens_f1'].cpu().numpy()
        e2e_stat[f'{mode_clf}']['avr_precision'] = log_dict[f'{mode}_ens_avr_precision'].cpu().numpy()
        e2e_stat[f'{mode_clf}']['aucroc'] = log_dict[f'{mode}_ens_aucroc'].cpu().numpy()
        

        pl_module.imputed_data_['data_dict'] = data_dict
        pl_module.imputed_data_['result_dict'] = result_dict
        pl_module.imputed_data_['e2e_stat'] = e2e_stat
        
        
        
            

    def impute_dataset(self, dataset, MASK, imp_ens_collector, num_idx, cat_idx):
        X_imputed = dataset.features.copy()
        y_ = dataset.labels
        #rmse, mae, acc = 0, 0, 0
        if len(num_idx)>0:
            data_n = dataset.features_clean[:, num_idx]
            mask_n = MASK[:, num_idx]

            row, col = (mask_n == 0).nonzero().numpy().T
            row_col = list(zip(row,col))
            row_col_pred = imp_ens_collector['num'].keys()

            assert set(row_col) == set(row_col_pred)
            #target = data_n[row, col]
            preds = np.array([imp_ens_collector['num'][key] for key in row_col])
            # rmse = rmse_f(preds, target)
            # mae = np.abs(preds - target).mean()


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

            #target = data_c[row, col]
            preds = np.array([imp_ens_collector['cat'][key] for key in row_col])
            #acc = (target == preds).sum()/len(target)

            # Impute dataset
            data_imp = X_imputed[:, cat_idx].copy()
            data_imp[row, col] = preds
            X_imputed[:, cat_idx] = data_imp
        
        return X_imputed, y_
        
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

def compute_imp_metrics(X_clean, X_imp, mask, num_idx, cat_idx):
    assert X_clean.shape == X_imp.shape
    assert X_clean.shape == mask.shape
    rmse, mae, acc = 0, 0, 0
    if len(num_idx) > 0:
        X_clean_n = X_clean[:, num_idx]
        X_imp_n = X_imp[:, num_idx]
        mask_n = mask[:, num_idx]

        row, col = (mask_n == 0).nonzero()
        target = X_clean_n[row, col]
        preds = X_imp_n[row, col]
        rmse = rmse_f(preds, target)
        mae = np.abs(preds - target).mean()
    if len(cat_idx) > 0:
        X_clean_c = X_clean[:, cat_idx]
        X_imp_c = X_imp[:, cat_idx]
        mask_c = mask[:, cat_idx]

        row, col = (mask_c == 0).nonzero()
        target = X_clean_c[row, col]
        preds = X_imp_c[row, col]
        acc = (target == preds).sum()/len(target)

    return rmse, mae, acc



# X_freq = dataset.features_freq
        # rmse_stat, mae_stat, acc_stat = compute_imp_metrics(X_clean=X_clean, 
        #     X_imp=X_freq,
        #     mask=mask_init,
        #     num_idx=num_idx,
        #     cat_idx=cat_idx
        #     )
        # log_dict["test_imp_rmse_(stat)"] = rmse_stat
        # log_dict["test_imp_mae_(stat)"] = mae_stat
        # log_dict["test_imp_acc_(stat)"] = acc_stat