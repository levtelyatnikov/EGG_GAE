import torch
import torchmetrics
from torch.nn import functional as F


class MetricCalculator():
    def __init__(self, num_classes, device=None):
        self.acc = torchmetrics.Accuracy().to(device)
        self.aucroc = torchmetrics.AUROC(num_classes=num_classes).to(device)
        self.f1 = torchmetrics.F1Score(num_classes=num_classes, average='weighted').to(device)
        self.avr_precision = torchmetrics.AveragePrecision(num_classes=num_classes).to(device)

    
    def calculate_metrics(self, labels, probs=None, logits=None, preds=None, sufix='ens'):
        if preds==None:
            assert logits != None, "Provide logits to get probabilities"

            preds = torch.argmax(logits, dim=1).detach().type(torch.int64).to(self.acc.device)
        else:
            preds = preds.type(torch.int64).to(self.acc.device)

        if probs==None:
            assert logits != None, "Provide logits to get probabilities"
            
            probs = F.softmax(logits, dim=1).detach()
            preds = torch.argmax(probs.detach(), dim=1)
        
        labels = labels.detach()

        # probs metrics
        avr_precision = self.avr_precision(probs, labels)
        aucroc = self.aucroc(probs, labels)
        
        # preds metrics
        accuracy = self.acc(preds, labels.to(self.acc.device))
        f1 = self.f1(preds, labels.to(self.f1.device))

        return {f"{sufix}_accuracy":accuracy,
                f"{sufix}_f1":f1,
                f"{sufix}_avr_precision":avr_precision,
                f"{sufix}_aucroc":aucroc}
    

