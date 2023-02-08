import os
import warnings
# hydra
import hydra 
from omegaconf import DictConfig, OmegaConf
import torch
# pytorch-lightning related imports
from pytorch_lightning import Trainer
import pytorch_lightning.loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from method import LitModel
from inference.val_logic import ValCallback
from inference.test_logic import TestCallback

from data_prep.pipeline import data_pipeline
from utils import config_preprocess, get_dataloader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

def setup_cuda(cfg: DictConfig):

    print("DEVICE COUNT: ",torch.cuda.device_count())
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.trainer.cuda_number
    
    print("DEVICE COUNT: ",torch.cuda.device_count())

@hydra.main(config_path='./configs', config_name='defaults')
def main(cfg: DictConfig):
    setup_cuda(cfg)
    print(OmegaConf.to_yaml(cfg))
    
    dataloader_cfg = cfg.dataloader
    dataset_name = dataloader_cfg.dataset_name
    experiment_name = dataloader_cfg.experiment_name
    val_size = dataloader_cfg.val_size
    data_split_seed = dataloader_cfg.data_seed
    miss_algo = dataloader_cfg.miss_algo
    def_fill_val = dataloader_cfg.imputation.fill_val
    p_miss = dataloader_cfg.imputation.init_corrupt 
    miss_seed = dataloader_cfg.imputation.seed 
    scale_type = dataloader_cfg.imputation.scale_type

    # Prepare data
    data_pipeline(dataset_name=dataset_name,
                  experiment_name=experiment_name,
                  def_fill_val=def_fill_val,
                  p_miss=p_miss, # initial data corruption
                  miss_algo=miss_algo,
                  miss_seed=miss_seed, 
                  val_size=val_size,
                  data_split_seed=data_split_seed,
                  scale_type=scale_type
                  )

    cfg.wadb.logger_name = f'{cfg.dataloader.dataset_name}_{cfg.model.edge_generation_type}_k(noSKIPCON)'
    
    # # # Configure weight and biases 
    logger = pl_loggers.WandbLogger(
        project=cfg.wadb.logger_project_name,
        name=cfg.wadb.logger_name if cfg.wadb.logger_name != 'None' else None, 
        entity=cfg.wadb.entity,
        
        )

    # --- Callbacks ---
    checkpoint_callback = ModelCheckpoint(
        monitor='val_ens_aucroc',
        mode='max',
        save_top_k=1,
        save_last=False,
        verbose=True,
        dirpath="checkpoints",
        filename="epoch_{epoch:03d}",
    )
    early_stopping = EarlyStopping(monitor="val_ens_aucroc", mode="max", min_delta=0.00, patience=3)

    # Setup dataloader and model
    
    datamodule = get_dataloader(cfg)

    # Get additional statistics from dataset
    cfg = config_preprocess(cfg, datamodule)

    callbacks = [ValCallback(num_classes=cfg.model.outsize, device='cuda:0'),
                 TestCallback(num_classes=cfg.model.outsize, device='cuda:0'), 
                 LearningRateMonitor("step"), checkpoint_callback, early_stopping]

    # Configure trained
    trainer = Trainer(gpus=1, accelerator='gpu', 
        logger=logger if cfg.trainer.is_logger_enabled else False,
        num_sanity_val_steps=-1, 
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        max_epochs=cfg.model.opt.max_epochs,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        callbacks=callbacks if cfg.trainer.is_logger_enabled else [])

    
    
    model = LitModel(datamodule=datamodule, cfg=cfg)
    
    # Train
    trainer.fit(model, datamodule)

    
    trainer.test(model=model, test_dataloaders=datamodule.test_dataloader(), ckpt_path="best")
    
    #trainer.test(model=model, test_dataloaders=datamodule.test_dataloader(), ckpt_path="best")
    
    print('Training is done!')
    

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fxn()

    main()
