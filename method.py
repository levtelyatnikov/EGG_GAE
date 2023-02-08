import torch
import pytorch_lightning as pl
from omegaconf import DictConfig
from models.model import Network
from models.edge_generation.EGmodule import EdgeGenerationModule
from utils import ModelTraverse 


def TauSetup(module, **args):
    module.cfg = args['cfg']                       

def TauUpdate(module, **args):
    module.current_step = args['global_step'] 
        
def init_weights(module, **args):
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.xavier_uniform(module.weight)
        module.bias.data.fill_(0.01)
    
class LitModel(pl.LightningModule):
    def __init__(self, datamodule, cfg: DictConfig):
        """
        datamodule: datamodule (lightning)
        cfg: config file, 
        - consist of dataloader, model, 
        - trainer, imputation parameters
        
        """
        super().__init__()
        
        # Save pytorch lightning parameters   
        # - save configs + self.hparams=cfg
        self.save_hyperparameters(cfg)
        pl.utilities.seed.seed_everything(self.hparams.seed_everything)
       
        # Get model from .yaml file
        self.cfg=cfg
        self.model = Network(cfg=self.hparams, train_dataloader=datamodule.train_dataloader())
        self.imputed_data_ = {}

        self.datamodule = datamodule
        # Setup Tau self.hparams
        ModelTraverse(model=self.model, 
                     SearchModule=EdgeGenerationModule,
                     func=TauSetup, cfg=self.hparams
                     )

        self.model_initialization()
        self.max_f1 = torch.Tensor([-1])
        self.infromation_keeper = {}

    def model_initialization(self):
        ModelTraverse(model=self.model, 
                     SearchModule=EdgeGenerationModule,
                     func=init_weights)
        
    # Logic for a single training step
    def training_step(self, batch, batch_idx):
        train_loss, _, _ = self.model.loss_function(batch)
        logs = {key: val.item() for key, val in train_loss.items()}
        self.log_dict(logs) # sync_dist=True
        return train_loss

    # logic for a single validation step
    def validation_step(self, batch, batch_idx):
        val_loss, preds, probs, num_rec, cat_outputs  = self.model.loss_function(batch, mode='val')
        return val_loss, preds, probs , num_rec, cat_outputs
    

    def test_step(self, batch, batch_idx):
        test_loss, preds, probs, num_rec, cat_outputs  = self.model.loss_function(batch, mode='test')
        return test_loss, preds, probs, num_rec, cat_outputs
        
    def configure_optimizers(self): 
        
        if self.hparams.model.opt.optimizer == 'RMSPROP':
            optimizer = torch.optim.RMSprop(self.model.parameters(), 
                            lr = self.hparams.model.opt.lr, 
                            weight_decay = self.hparams.model.opt.weight_decay,
                            momentum=self.hparams.model.opt.momentum)
                            
        elif self.hparams.model.opt.optimizer == 'SGD':
            optimizer = torch.optim.SGD(self.model.parameters(),
                            lr = self.hparams.model.opt.lr, 
                            weight_decay = self.hparams.model.opt.weight_decay,
                            momentum=self.hparams.model.opt.momentum)
        else: 
            optimizer = torch.optim.AdamW(self.model.parameters(), 
                            lr = self.hparams.model.opt.lr, 
                            weight_decay = self.hparams.model.opt.weight_decay,)
                            #momentum=self.hparams.model.opt.momentum)

        warmup_steps_pct = self.hparams.model.opt.warmup_steps_pct
        decay_steps_pct = self.hparams.model.opt.decay_steps_pct
        total_steps = self.hparams.model.opt.max_epochs * self.hparams.model.opt.loader_batches #len(self.datamodule.train_dataloader())

        def warm_and_decay_lr_scheduler(step: int):
            warmup_steps = warmup_steps_pct * total_steps
            decay_steps = decay_steps_pct * total_steps
            assert step <= total_steps
            if step < warmup_steps:
                factor = step / warmup_steps
            else:
                factor = 1
            factor *= self.hparams.model.opt.scheduler_gamma ** (step / decay_steps)
            return factor

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=warm_and_decay_lr_scheduler)

        return (
            [optimizer],
            [{"scheduler": scheduler, "interval": "step",}],
        )

