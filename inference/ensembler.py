import torch
import numpy as np
from metrics.metrics import MetricCalculator 


class Ensemble():
    "Main idea of this class is to output predictions"
    def __init__(self, num_classes, device):
        self.device = device
        self.metric_calculator = MetricCalculator(num_classes=num_classes)

    def proscess(self, pred_collector, k=None, imp_collector=None):
        # idx prediction mapper
        mapper = dict(pred_collector['mapper'].items())
        
        # Ensemble
        ensemble = self.ensemble(dict_probs=pred_collector['probs'],
                                            dict_pred=pred_collector['preds'],
                                            mapper=mapper, k=k)

        # Ensembel
        labels=torch.Tensor([i[0] for i in ensemble]).to(self.device).type(torch.int64)
        preds=torch.Tensor([i[1] for i in ensemble]).to(self.device)
        probs=torch.cat([i[2] for i in ensemble], dim=0).to(self.device)

        # Calculate metrics
        pred_ens_collector = self.metric_calculator.calculate_metrics(labels=labels,
                                                            preds=preds,
                                                            probs=probs)
                                   
        pred_ens_collector['min_predictions'] = np.min([len(val) for val in pred_collector['preds'].values()])
        pred_ens_collector['mean_predictions'] = np.mean([len(val) for val in pred_collector['preds'].values()])

        if imp_collector is not None:
            imp_ens_collector = self.ensemble_imput(imp_collector, k=k)
        

        assert (k != None) and (k >= 1)
        
        return pred_ens_collector, imp_ens_collector

    def ensemble_imput(self, imp_collector, k=None):
        ensemble = {'num':{}, 'cat':{}}
        
        num_keys = imp_collector['num'].keys()
        if len(num_keys) > 0:
            for key in num_keys:
                ensemble['num'][key] = np.mean([val.cpu().numpy() for val in imp_collector['num'][key]][:k])

        cat_keys = list(imp_collector['cat'].keys())
        if len(cat_keys) > 0:
            for key in cat_keys:
                logits = torch.concat(list(map(lambda x: x.unsqueeze(0),imp_collector['cat'][key])), dim=0)[:k].mean(0)
                pred = torch.argmax(logits, dim=-1)
                ensemble['cat'][key] = pred.cpu().numpy()

        return ensemble

    def ensemble(self, dict_probs, dict_pred, mapper, k=None):
        ensemble = []
        
        assert dict_probs.keys() == mapper.keys()
        assert dict_pred.keys() == mapper.keys()

        for unique_idx, label in mapper.items():
            label = label.type(torch.int64)
            # Calculate ensemble
            # Get first k probs
            probability = torch.cat(dict_probs[unique_idx][:k], dim=0).mean(0, keepdim=True)
            prediction = torch.argmax(probability)
            ensemble.append((label, prediction, probability))

        return ensemble
    
