import pytorch_lightning as pl
from omegaconf.dictconfig import DictConfig
from torch_geometric.data import LightningDataset
from datasets.datasets import TrainImputeDataset, TestImputeDataset
from datasets.upload_data import load_prepared_data


class PL_DataModule(pl.LightningDataModule):
    """
    
    This class creates train/validation datasets
    In usual case this class do not need to be changed
    all data manipulations are performed in dataset.py and 
    transformfactory.py files
    """
    def __init__(
        self,
        cfg: DictConfig, model_type, data
    ):
        super().__init__()
        
                
        self.cfg = cfg
        if data == None:
            data = load_prepared_data(cfg.dataset_name, cfg.experiment_name)
        else:
            # data is uploaded in advance
            pass 
            
        train_dataset = TrainImputeDataset(
            split='train',
            cfg=cfg,
            features=data["X_train_deg_freq"], # during training test values are substituted with statistics
            labels=data["y_train"],
            features_clean=data['X_train_clean'],
            MASK_init=data["mask_init_train"],
            num_idx=data["num_idx"],
            NumFillWith=data["NumFillWith"],
            cat_idx=data["cat_idx"], 
            CatFillWith=data["CatFillWith"], 
            cat_dims=data["cat_dims"],
            MostFreqClass=data["MostFreqClass"], 
            model_type=model_type
            )
            
        # val_dataset = ValDataset(features=data["X_val_deg_freq"],
        #                          mask_init="mask_mcar_val"),
        #                          features_clean=data["X_val_deg_freq"]) 
        
        # test_dataset = TestDataset(features=data["X_test_deg_freq"], 
        #                            mask_init="mask_init_test"), 
        #                            features_clean=data["X_test_clean"])  
         
        val_dataset = TestImputeDataset( #ValImputeDataset(
            split='val',
            cfg=cfg,
            features=data["X_val_deg_freq"], # During validation test values are substituted with statistics
            labels=data["y_val"], 
            features_clean=data["X_val_clean"],
            MASK_init=data["mask_init_val"],

            num_idx=data["num_idx"], 
            NumFillWith=data["NumFillWith"],
            cat_idx=data["cat_idx"], 
            CatFillWith=data["CatFillWith"], 
            cat_dims=data["cat_dims"],
            MostFreqClass=data["MostFreqClass"], 
            model_type=model_type)

        # Train test split
        test_dataset = TestImputeDataset( 
            split='test', 
            cfg=cfg,
            features=data["X_test_deg"], 
            features_freq=data["X_test_deg_freq"],
            labels=data["y_test"],
            features_clean=data["X_test_clean"], 
            MASK_init=data["mask_init_test"], 
            num_idx=data["num_idx"], 
            NumFillWith=data["NumFillWith"],
            cat_idx=data["cat_idx"], 
            CatFillWith=data["CatFillWith"], 
            cat_dims=data["cat_dims"],
            MostFreqClass=data["MostFreqClass"],
            model_type=model_type
            )
        
        # # Valid test split
        # val_test_dataset = TestImputeDataset(
        #     split='test',
        #     cfg=cfg,
        #     features=data["X_val_deg"],
        #     labels=data["y_val"], 
        #     features_clean=data["X_val_clean"],
        #     features_freq=data["X_val_deg_freq"],
        #     MASK_init=data["mask_init_val"],

        #     num_idx=data["num_idx"], 
        #     NumFillWith=data["NumFillWith"],
        #     cat_idx=data["cat_idx"], 
        #     CatFillWith=data["CatFillWith"], 
        #     cat_dims=data["cat_dims"],
        #     MostFreqClass=data["MostFreqClass"], 
        #     model_type=model_type)
        
        # Valid test split
        train_test_dataset = TestImputeDataset(
            split='train_test',
            cfg=cfg,
            features=data["X_train_deg"],
            labels=data["y_train"], 
            features_clean=data["X_train_clean"],
            features_freq=data["X_train_deg_freq"],
            MASK_init=data["mask_init_train"],

            num_idx=data["num_idx"], 
            NumFillWith=data["NumFillWith"],
            cat_idx=data["cat_idx"], 
            CatFillWith=data["CatFillWith"], 
            cat_dims=data["cat_dims"],
            MostFreqClass=data["MostFreqClass"], 
            model_type=model_type)


        self.dl = LightningDataset(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            num_workers=cfg.num_workers,
            pin_memory=False, 
            batch_size=1
            )
        # This it needed only to provide 
        self.dl2 = LightningDataset(
            train_dataset=train_test_dataset,
            val_dataset=train_test_dataset,
            test_dataset=test_dataset,
            num_workers=cfg.num_workers,
            pin_memory=False, 
            batch_size=1
            )
                                   
        self.data = data
        
    def get_all_data(self):
        return self.data

    def train_dataloader(self):
        return self.dl.train_dataloader()

    def val_dataloader(self):
        return self.dl.val_dataloader() 
        
    def test_dataloader(self):
        return self.dl.test_dataloader() 
    
    def train_test_dataloader(self):
        return self.dl2.train_dataloader()

