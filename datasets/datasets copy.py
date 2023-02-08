import torch
import numpy as np
from torch_geometric.data import Data
import numpy as np
import random 
from omegaconf.dictconfig import DictConfig
from torch_geometric.data import Dataset
from abc import abstractmethod

class ImputationBase(Dataset):
    """Dataset example"""
    def __init__(self, features, labels, features_clean, MASK_init,
                 split, cfg, num_idx, NumFillWith,
                 cat_idx, CatFillWith, cat_dims, MostFreqClass, 
                 model_type, features_freq=None):
    
        super().__init__() #root='None', transform=None, pre_transform=None, pre_filter=None
        """Initialize

        cfg:
         :data_dir: data directory
         :transforms: TransformObject Class which is defined in the transformfactory.py file
         :target_transforms: TransformLabel Class which is defined in the transformfactory.py file

         :split: train/val split
         :val_size: validation size
         :seed: seed
        """

        self.cfg, self.split = cfg, split

        self.features = features
        self.labels = labels
        self.setup_batchsize()

    

        """

        features: features of the dataset
        labels: labels of the dataset
        features_clean: 
        MASK_init: mask, 1 - not corrupted, 0 - corrupted
        split: split
        cfg: configs
        num_idx: numerical indexes
        NumFillWith: numerical values to fill with when corrupting 
        cat_idx: categorical indexes
        CatFillWith: categorical values to fill with when corrupting
        cat_dims: categorical dimensions
        """
        # Convert to torch
        self.features_clean = features_clean
        self.features_freq = features_freq
        # Check if dataset consist of numerical fetures
        if len(num_idx) > 0:
            self.num_idx = torch.tensor(num_idx).type(torch.int64)
            self.NumFillWith = torch.tensor(NumFillWith).type(torch.float32)
        else:
            self.num_idx, self.NumFillWith = torch.tensor([]).type(torch.int64), \
                                             torch.tensor([]).type(torch.float32)

        # Check if datasets consist of categorical features
        if len(cat_idx) > 0:
            self.cat_dims = torch.tensor(cat_dims).type(torch.int64)
            self.cat_idx = torch.tensor(cat_idx).type(torch.int64)
            self.CatFillWith = torch.tensor(CatFillWith).type(torch.float32)
            self.MostFreqClass = torch.tensor(MostFreqClass).type(torch.float32)
            self.mappers = []
            for dim in self.CatFillWith:
                mapper = dict(zip(np.arange(dim), np.arange(dim)))
                mapper[np.float(self.cfg.imputation.fill_val)] = dim.item()
                mapper[np.float(dim.item())] = dim.item()
                self.mappers.append(mapper)

        else:
            self.cat_idx, self.CatFillWith = torch.tensor([]).type(torch.float32), \
                                             torch.tensor([]).type(torch.float32)
            self.MostFreqClass = torch.tensor(MostFreqClass).type(torch.float32)
            
            assert len(cat_dims)==0
            self.cat_dims = torch.tensor([]).type(torch.int64)
            
        
        # Mask to torch tensor
        self.MASK_init = torch.tensor(MASK_init).type(torch.float32)
        self.missingness = self.cfg.fly_corrupt
        assert 0.0 <= self.missingness <= 1.0, "check assert 0.0 <= self.missingness <= 1.0 {self.missingness}"
    
        assert len(cat_idx) == len(cat_dims)
        self.dataset_init_dim = len(self.cat_dims) + len(self.num_idx)

        self.n_engrances = self.graph_size * self.dataset_init_dim
        self.corrupt_size = int(self.missingness * self.n_engrances)
        #self.batch_flatten_idx = torch.tensor(list(range(self.n_engrances))).type(torch.float64)
        
        self.len_cat_idx = len(self.cat_idx)
        self.len_num_idx = len(self.num_idx)
        self.model_type = model_type

        # batch sampling 
        if split == 'train':
            self.batch_counter = 0
            self.mix_data()
    
        if split in ['test', 'val']:
            min_mean_predictions = self.cfg.min_sample_predictions
            # NN always gives the same logits for the same input
            # if self.model_type == 'NN_module':
            #     min_mean_predictions = 1

            # Make sure to have same sequence of batches
            np.random.seed(self.cfg.val_seed)
            # Stable index needed to perform validation
            # over the same batches during val/test
            self.stable_idx = []
            
            flag = True
            while flag:
                one_pass = self.mix_data(return_idx=True)
                self.stable_idx.append(one_pass)
                # Check if every datum appears at least min_mean_predictions times
                # in the batches
                if np.min(np.unique(self.stable_idx, return_counts=True)[1]) >= min_mean_predictions:
                    flag = False

            assert len(set(np.unique(self.stable_idx)).intersection(set(self.idxs))) == len(self.idxs)
            self.stable_idx = np.concatenate(self.stable_idx, axis=0)
            
            self.n_batches = len(self.stable_idx)
            self.idxs = np.sort(self.idxs)
            
            if split=='val':
                self.prepare_validation_MCAR_once()

    
    def setup_batchsize(self,):
   
        if self.split == "train":
            self.graph_size = self.cfg.train_graph_size

        if self.split == "val":
            self.graph_size = self.cfg.val_graph_size

        if self.split == "test":
            self.graph_size = self.cfg.val_graph_size
                
        # Setup available indexes
        self.idxs = np.arange(self.features.shape[0])
    

    def get(self, idx):
        """Return image and label"""
        raise NotImplementedError("This method has to be implemented for Train Val separetly")

        
    def len(self):
        return self.n_batches


    def mix_data(self, return_idx=False):
        lenidxs = len(self.idxs)
        if return_idx==False:
            # if return_idx == False
            # write one pass dataset directly into self.stable_idx
            self.stable_idx = []
        
        np.random.shuffle(self.idxs)
        if (lenidxs % self.graph_size) == 0:
            one_pass = self.idxs.reshape(-1, self.graph_size)
        else: 
            first_vals = (lenidxs // self.graph_size) * self.graph_size
            one_pass = self.idxs[:first_vals].reshape(-1, self.graph_size)        
            tail = np.concatenate([self.idxs[first_vals:],\
                                self.idxs[:self.graph_size - (lenidxs % self.graph_size)]]).reshape(-1, self.graph_size)

            # batched dataset pass
            one_pass = np.concatenate([one_pass, tail], axis=0)
        
        if return_idx==False:
            self.stable_idx.append(one_pass)

            # Make sure to pass the whole dataset
            assert len(set(np.unique(self.stable_idx)).intersection(set(self.idxs))) == len(self.idxs)
            self.stable_idx = np.concatenate(self.stable_idx, axis=0)
            
            self.n_batches = len(self.stable_idx)
            self.idxs = np.sort(self.idxs)
        else:
            self.idxs = np.sort(self.idxs)
            return one_pass  
    
    def extract_batch(self, idxs, seed_idx=0, mode="train"):
        if mode in ['train', 'test']:
            # "Sampling" from dataset
            features= self.features[idxs]
                                   
        
            # mask of test values
            mask_init = self.MASK_init[idxs] 
        else: 
            assert mode=='val'
            features_val_mcar = self.features_mcar[idxs]
            mask_arti_mcar = self.MASK_val[idxs] 
        
        features_clean =  torch.tensor(self.features_clean[idxs]).type(torch.float32)
        label = self.labels[idxs]

        if mode=="train":
            # Introduce MCAR noise during training and validation
            # ----------------------------------------------------
            # During traning because the initial noise can be:
            # MCAR, MAR, MNAR, however we do not know 
            # the true noise distribution, so we assume MCAR always.
            # The test values are substituted with the most frequent class
            # or mean/median value estimated from non-corrupted values of
            # the corrupted training dataset (the one whoch we loaded).
            # We calculate the statistics only on training part 
            # of the dataset "loaded" dataset.
            # In this way we can introduce artificial noise during training
            # which allow us to train the model on MCAR noise (MCAR noise 
            # can be introduced into non-corrupted values of the dataset).
            # so test values do not appear in the loss function.
            # ----------------------------------------------------
            # For the validation we apply same schema, such that we can
            # how well our model reconstructs values which are corrupted 
            # with MCAR noise.
            
            # During validation because the initial noise is MCAR
            features_train, mask_arti = self.batch_preprocessing(X=features, mask_init=mask_init) 
            features_train = torch.tensor(features_train).type(torch.float32)
            
            return features_clean, features_train, None, label,\
                    None, mask_arti
        elif mode=='test':
            features = self.batch_preprocessing(X=features, mask_init=mask_init) 
            features_test = torch.tensor(features).type(torch.float32)

            return  features_clean, None, features_test, label,\
                    mask_init, None
        elif mode == 'val':
            features_val_mcar = self.batch_preprocessing(X=features_val_mcar, mask_init=mask_arti_mcar) 
            features_val_mcar = torch.tensor(features_val_mcar).type(torch.float32)
            
            return features_clean, features_val_mcar, None, label,\
                    None, mask_arti_mcar

    
    @abstractmethod
    def batch_preprocessing(self,  X, mask_init):
        # self.batch_to_dense
        NotImplementedError("batch_preprocess not implemented")
    
    def batch_to_dense(self, X, mask):
        # Map new corrupted categorical values to appropriate label, hence later it can be transformed into DenseEmb
        if self.len_cat_idx > 0:
            for mapper, col in zip(self.mappers, self.cat_idx):
                corrupted_idx = np.where(mask[:, col.numpy()] == 0)[0]
                X[corrupted_idx, col] = np.array(list(map(lambda x: int(mapper[x]),
                                                            X[corrupted_idx, col])))
        if self.len_num_idx > 0: 
            # FullFill new empty numerical values with mean of the estimated
            for col, val in zip(self.num_idx, self.NumFillWith):
                corrupted_idx = np.where(mask[:, col] == 0)[0]
                X[corrupted_idx, col] = val.item()
        return X
    
        
    
class TrainImputeDataset(ImputationBase):
    def get(self, idx):
        """
        idx: batch index
        """
        if self.batch_counter==0: 
            self.mix_data()
        
        idxs = self.stable_idx[idx]  
        #
        
        features_clean, features_train,\
        _, label,\
        _, mask_arti = self.extract_batch(idxs)
        
        # x - features that has been corrupted artificially and initially
        # x_init - fully clean features
        # x_clean_deg - features which were corrupted only initially
        data = Data(x=features_train, mask=mask_arti,
                    x_clean = features_clean,
                    y=torch.tensor(label).type(torch.int64),    
                    num_idx=self.num_idx, cat_idx=self.cat_idx,
                    cat_dims=self.cat_dims, NumFillWith=self.NumFillWith,
                    CatFillWith=self.CatFillWith, 
                    MostFreqClass=self.MostFreqClass,
                    unique_idx = torch.tensor(idxs).type(torch.int64), idx=idx,
                    bc=self.batch_counter, mode=torch.tensor([0]).type(torch.int64))

        
        # Shuffle train dataset during training.
        self.batch_counter += 1
        if self.batch_counter == (self.n_batches*int(self.cfg.num_workers)):
            self.batch_counter=0

        return data
    
    def batch_preprocessing(self, X, mask_init):
        
        """
        Inputs:
            dataset to corrupt
            % of data to eliminate[0,1]
            rand random state
            replace with = 'zero' or 'nan'
        Outputs:
            corrupted Dataset 
            binary mask
        """
        X_1d = X.flatten()
        mask_1d = torch.ones(X_1d.shape[0])#torch.ones(self.graph_size*self.dataset_init_dim)

        if self.corrupt_size==0:
            # Base case (not corrupt dataset)
            return X, mask_1d.reshape(X.shape)

        # Take only not test indices i.e. indices with 0 in mask_init
        available_idx = (mask_init.flatten() == 1).nonzero(as_tuple=True)[0] # self.batch_flatten_idx[mask_init.flatten() == 1]
        
        # Randomly select indices to corrupt
        p = torch.ones(available_idx.shape)
        p = p / p.sum()
        temp_idx = torch.multinomial(p, num_samples=self.corrupt_size, replacement=False)
        corrupt_idx = available_idx[temp_idx].type(torch.int64)
        
        X_1d[corrupt_idx] = self.cfg.imputation.fill_val
        mask_1d[corrupt_idx] = 0
        
        X_deg = X_1d.reshape(X.shape)
        mask_arti = mask_1d.reshape(X.shape)

        # Map new corrupted categorical values to appropriate label, hence later it can be transformed into DenseEmb
        X_deg = self.batch_to_dense(X_deg, mask_arti)
        
        return X_deg, mask_arti
    
    
    

class ValImputeDataset(ImputationBase):
    def get(self, idx):
        """
        idx: batch index
        """
    
        torch.manual_seed(int(idx * self.cfg.val_seed))
        np.random.seed(int(idx * self.cfg.val_seed))
        random.seed(int(idx * self.cfg.val_seed))
        
        idxs = self.stable_idx[idx]  
        #
        
        features_clean, features_val_mcar,\
        _, label,\
        _, mask_arti = self.extract_batch(idxs, mode="val")
        
        # x - features that has been corrupted artificially and initially
        # x_init - fully clean features
        # x_clean_deg - features which were corrupted only initially
        data = Data(x=features_val_mcar, mask=mask_arti,
                    x_clean = features_clean,
                    y=torch.tensor(label).type(torch.int64),    
                    num_idx=self.num_idx, cat_idx=self.cat_idx,
                    cat_dims=self.cat_dims, NumFillWith=self.NumFillWith,
                    CatFillWith=self.CatFillWith, 
                    MostFreqClass=self.MostFreqClass,
                    unique_idx = torch.tensor(idxs).type(torch.int64), idx=idx,
                    mode=torch.tensor([0]).type(torch.int64))

        
      
        return data
        
    def prepare_validation_MCAR_once(self, ):
        torch.manual_seed(int(self.cfg.val_seed))
        np.random.seed(int(self.cfg.val_seed))
        random.seed(int(self.cfg.val_seed))
        
        # We have to degrage the whole val dataset once and keep it
        self.features_mcar, self.MASK_val = self.data_val_preprocessing(
                                                self.features,
                                                self.MASK_init
                                                )
        
    def batch_preprocessing(self,  X, mask_init):
        return self.batch_to_dense(X, mask_init)
        
    def data_val_preprocessing(self, X, mask_init):
        
        """
        Inputs:
            dataset to corrupt
            % of data to eliminate[0,1]
            rand random state
            replace with = 'zero' or 'nan'
        Outputs:
            corrupted Dataset 
            binary mask
        """
        X_1d = X.flatten()
        mask_1d = torch.ones(X_1d.shape[0])#torch.ones(self.graph_size*self.dataset_init_dim)

        if self.corrupt_size==0:
            # Base case (not corrupt dataset)
            return X, mask_1d.reshape(X.shape)

        # Take only not test indices i.e. indices with 0 in mask_init
        available_idx = (mask_init.flatten() == 1).nonzero(as_tuple=True)[0] # self.batch_flatten_idx[mask_init.flatten() == 1]
        
        # Randomly select indices to corrupt
        p = torch.ones(available_idx.shape)
        p = p / p.sum()
        temp_idx = torch.multinomial(p, num_samples=self.corrupt_size, replacement=False)
        corrupt_idx = available_idx[temp_idx].type(torch.int64)
        
        X_1d[corrupt_idx] = self.cfg.imputation.fill_val
        mask_1d[corrupt_idx] = 0
        
        X_deg = X_1d.reshape(X.shape)
        mask_arti = mask_1d.reshape(X.shape) 
        return X_deg, mask_arti

        

class TestImputeDataset(ImputationBase):
    def get(self, idx):
        
        idxs = self.stable_idx[idx] 

        features_clean, _,\
        features_test, label,\
        mask_init, _ = self.extract_batch(idxs, seed_idx=idx, mode="test")

        # x - features that has been corrupted artificially and initially
        # x_clean - fully clean features
        # x_initCorr - features which were corrupted only initially
        data = Data(x=features_test, mask=mask_init, 
                    x_clean = features_clean,
                    y=torch.tensor(label).type(torch.int64),
                    num_idx=self.num_idx, cat_idx=self.cat_idx,
                    cat_dims=self.cat_dims, NumFillWith=self.NumFillWith,
                    CatFillWith=self.CatFillWith, 
                    MostFreqClass=self.MostFreqClass, 
                    unique_idx = torch.tensor(idxs).type(torch.int64),
                    mode=torch.tensor([2]).type(torch.int64))


        return data

    def batch_preprocessing(self,  X, mask_init):
        return self.batch_to_dense(X, mask_init)
 



# import torch
# import numpy as np
# from torch_geometric.data import Data
# import numpy as np
# import random 
# from omegaconf.dictconfig import DictConfig
# from torch_geometric.data import Dataset
# from abc import abstractmethod

# class CustomDataset(Dataset):
#     """Dataset example"""
#     def __init__(self, features, labels, split, cfg: DictConfig):
#         super().__init__(root='None', transform=None, pre_transform=None, pre_filter=None)
#         """Initialize

#         cfg:
#          :data_dir: data directory
#          :transforms: TransformObject Class which is defined in the transformfactory.py file
#          :target_transforms: TransformLabel Class which is defined in the transformfactory.py file

#          :split: train/val split
#          :val_size: validation size
#          :seed: seed
#         """

#         self.cfg, self.split = cfg, split

#         self.features = features
#         self.labels = labels
#         self.setup_batchsize()

#     def setup_batchsize(self,):
   
#         if self.split == "train":
#             self.graph_size = self.cfg.train_graph_size

#         if self.split == "val":
#             self.graph_size = self.cfg.val_graph_size

#         if self.split == "test":
#             self.graph_size = self.cfg.val_graph_size
                
#         # Setup available indexes
#         self.idxs = np.arange(self.features.shape[0])
    

#     def get(self, idx):
#         """Return image and label"""
#         raise NotImplementedError("This method has to be implemented for Train Val separetly")

        
#     def len(self):
#         return self.n_batches
    
   


# class ImputationBase(CustomDataset):
#     def __init__(self, features, labels, features_clean, MASK_init,
#                  split, cfg, num_idx, NumFillWith,
#                  cat_idx, CatFillWith, cat_dims, MostFreqClass, 
#                  model_type, features_freq=None):
#         super().__init__(features, labels, split, cfg)
#         """

#         features: features of the dataset
#         labels: labels of the dataset
#         features_clean: 
#         MASK_init: mask, 1 - not corrupted, 0 - corrupted
#         split: split
#         cfg: configs
#         num_idx: numerical indexes
#         NumFillWith: numerical values to fill with when corrupting 
#         cat_idx: categorical indexes
#         CatFillWith: categorical values to fill with when corrupting
#         cat_dims: categorical dimensions
#         """
#         # Convert to torch
#         self.features_clean = features_clean
#         self.features_freq = features_freq
#         # Check if dataset consist of numerical fetures
#         if len(num_idx) > 0:
#             self.num_idx = torch.tensor(num_idx).type(torch.int64)
#             self.NumFillWith = torch.tensor(NumFillWith).type(torch.float32)
#         else:
#             self.num_idx, self.NumFillWith = torch.tensor([]).type(torch.int64), \
#                                              torch.tensor([]).type(torch.float32)

#         # Check is datasets consist of categorical features
#         if len(cat_idx) > 0:
#             self.cat_dims = torch.tensor(cat_dims).type(torch.int64)
#             self.cat_idx = torch.tensor(cat_idx).type(torch.int64)
#             self.CatFillWith = torch.tensor(CatFillWith).type(torch.float32)
#             self.MostFreqClass = torch.tensor(MostFreqClass).type(torch.float32)
#             self.mappers = []
#             for dim in self.CatFillWith:
#                 mapper = dict(zip(np.arange(dim), np.arange(dim)))
#                 mapper[np.float(self.cfg.imputation.fill_val)] = dim.item()
#                 mapper[np.float(dim.item())] = dim.item()
#                 self.mappers.append(mapper)

#         else:
#             self.cat_idx, self.CatFillWith = torch.tensor([]).type(torch.float32), \
#                                              torch.tensor([]).type(torch.float32)
#             self.MostFreqClass = torch.tensor(MostFreqClass).type(torch.float32)
            
#             assert len(cat_dims)==0
#             self.cat_dims = torch.tensor([]).type(torch.int64)
            
        
#         # Mask to torch tensor
#         self.MASK_init = torch.tensor(MASK_init).type(torch.float32)
#         self.missingness = self.cfg.fly_corrupt
#         assert 0.0 <= self.missingness <= 1.0, "check assert 0.0 <= self.missingness <= 1.0 {self.missingness}"
    
#         assert len(cat_idx) == len(cat_dims)
#         self.dataset_init_dim = len(self.cat_dims) + len(self.num_idx)

#         self.n_engrances = self.graph_size * self.dataset_init_dim
#         self.corrupt_size = int(self.missingness * self.n_engrances)
#         #self.batch_flatten_idx = torch.tensor(list(range(self.n_engrances))).type(torch.float64)
        
#         self.len_cat_idx = len(self.cat_idx)
#         self.len_num_idx = len(self.num_idx)
#         self.model_type = model_type

#         # batch sampling 
#         if split == 'train':
#             self.batch_counter = 0
#             self.mix_data()
    
#         if split in ['test', 'val']:
#             min_mean_predictions = self.cfg.min_sample_predictions
#             # NN always gives the same logits for the same input
#             # if self.model_type == 'NN_module':
#             #     min_mean_predictions = 1

#             # Make sure to have same sequence of batches
#             np.random.seed(self.cfg.val_seed)
#             # Stable index needed to perform validation
#             # over the same batches during val/test
#             self.stable_idx = []
            
#             flag = True
#             while flag:
#                 one_pass = self.mix_data(return_idx=True)
#                 self.stable_idx.append(one_pass)
#                 # Check if every datum appears at least min_mean_predictions times
#                 # in the batches
#                 if np.min(np.unique(self.stable_idx, return_counts=True)[1]) >= min_mean_predictions:
#                     flag = False

#             assert len(set(np.unique(self.stable_idx)).intersection(set(self.idxs))) == len(self.idxs)
#             self.stable_idx = np.concatenate(self.stable_idx, axis=0)
            
#             self.n_batches = len(self.stable_idx)
#             self.idxs = np.sort(self.idxs)
            
#             if split=='val':
#                 self.prepare_validation_MCAR_once()

#     def len(self):
#         return self.n_batches
        
#     def mix_data(self, return_idx=False):
#         lenidxs = len(self.idxs)
#         if return_idx==False:
#             # if return_idx == False
#             # write one pass dataset directly into self.stable_idx
#             self.stable_idx = []
        
#         np.random.shuffle(self.idxs)
#         if (lenidxs % self.graph_size) == 0:
#             one_pass = self.idxs.reshape(-1, self.graph_size)
#         else: 
#             first_vals = (lenidxs // self.graph_size) * self.graph_size
#             one_pass = self.idxs[:first_vals].reshape(-1, self.graph_size)        
#             tail = np.concatenate([self.idxs[first_vals:],\
#                                 self.idxs[:self.graph_size - (lenidxs % self.graph_size)]]).reshape(-1, self.graph_size)

#             # batched dataset pass
#             one_pass = np.concatenate([one_pass, tail], axis=0)
        
#         if return_idx==False:
#             self.stable_idx.append(one_pass)

#             # Make sure to pass the whole dataset
#             assert len(set(np.unique(self.stable_idx)).intersection(set(self.idxs))) == len(self.idxs)
#             self.stable_idx = np.concatenate(self.stable_idx, axis=0)
            
#             self.n_batches = len(self.stable_idx)
#             self.idxs = np.sort(self.idxs)
#         else:
#             self.idxs = np.sort(self.idxs)
#             return one_pass  
    
#     def extract_batch(self, idxs, seed_idx=0, mode="train"):
#         if mode in ['train', 'test']:
#             # "Sampling" from dataset
#             features= self.features[idxs]
                                   
        
#             # mask of test values
#             mask_init = self.MASK_init[idxs] 
#         else: 
#             assert mode=='val'
#             features_val_mcar = self.features_mcar[idxs]
#             mask_arti_mcar = self.MASK_val[idxs] 
        
#         features_clean =  torch.tensor(self.features_clean[idxs]).type(torch.float32)
#         label = self.labels[idxs]

#         if mode=="train":
#             # Introduce MCAR noise during training and validation
#             # ----------------------------------------------------
#             # During traning because the initial noise can be:
#             # MCAR, MAR, MNAR, however we do not know 
#             # the true noise distribution, so we assume MCAR always.
#             # The test values are substituted with the most frequent class
#             # or mean/median value estimated from non-corrupted values of
#             # the corrupted training dataset (the one whoch we loaded).
#             # We calculate the statisrics only on training part 
#             # of the dataset "loaded" dataset.
#             # In this way we can introduce artificial noise during training
#             # which allow us to train the model on MCAR noise (MCAR noise 
#             # can be introduced into non-corrupted values of the dataset).
#             # so test values do not appear in the loss function.
#             # ----------------------------------------------------
#             # For the validation we apply same schema, such that we can
#             # how well our model reconstructs values which are corrupted 
#             # with MCAR noise.
            
#             # During validation because the initial noise is MCAR
#             features_train, mask_arti = self.batch_preprocessing(X=features, mask_init=mask_init) 
#             features_train = torch.tensor(features_train).type(torch.float32)
            
#             return features_clean, features_train, None, label,\
#                     None, mask_arti
#         elif mode=='test':
#             features = self.batch_preprocessing(X=features, mask_init=mask_init) 
#             features_test = torch.tensor(features).type(torch.float32)

#             return  features_clean, None, features_test, label,\
#                     mask_init, None
#         elif mode == 'val':
#             features_val_mcar = self.batch_preprocessing(X=features_val_mcar, mask_init=mask_arti_mcar) 
#             features_val_mcar = torch.tensor(features_val_mcar).type(torch.float32)
            
#             return features_clean, features_val_mcar, None, label,\
#                     None, mask_arti_mcar

    
#     @abstractmethod
#     def batch_preprocessing(self,  X, mask_init):
#         # self.batch_to_dense
#         NotImplementedError("batch_preprocess not implemented")
    
#     def batch_to_dense(self, X, mask):
#         # Map new corrupted categorical values to appropriate label, hence later it can be transformed into DenseEmb
#         if self.len_cat_idx > 0:
#             for mapper, col in zip(self.mappers, self.cat_idx):
#                 corrupted_idx = np.where(mask[:, col.numpy()] == 0)[0]
#                 X[corrupted_idx, col] = np.array(list(map(lambda x: int(mapper[x]),
#                                                             X[corrupted_idx, col])))
#         if self.len_num_idx > 0: 
#             # FullFill new empty numerical values with mean of the estimated
#             for col, val in zip(self.num_idx, self.NumFillWith):
#                 corrupted_idx = np.where(mask[:, col] == 0)[0]
#                 X[corrupted_idx, col] = val.item()
#         return X
    
        
    
# class TrainImputeDataset(ImputationBase):
#     def get(self, idx):
#         """
#         idx: batch index
#         """
#         if self.batch_counter==0: 
#             self.mix_data()
        
#         idxs = self.stable_idx[idx]  
#         #
        
#         features_clean, features_train,\
#         _, label,\
#         _, mask_arti = self.extract_batch(idxs)
        
#         # x - features that has been corrupted artificially and initially
#         # x_init - fully clean features
#         # x_clean_deg - features which were corrupted only initially
#         data = Data(x=features_train, mask=mask_arti,
#                     x_clean = features_clean,
#                     y=torch.tensor(label).type(torch.int64),    
#                     num_idx=self.num_idx, cat_idx=self.cat_idx,
#                     cat_dims=self.cat_dims, NumFillWith=self.NumFillWith,
#                     CatFillWith=self.CatFillWith, 
#                     MostFreqClass=self.MostFreqClass,
#                     unique_idx = torch.tensor(idxs).type(torch.int64), idx=idx,
#                     bc=self.batch_counter, mode=torch.tensor([0]).type(torch.int64))

        
#         # Shuffle train dataset during training.
#         self.batch_counter += 1
#         if self.batch_counter == (self.n_batches*int(self.cfg.num_workers)):
#             self.batch_counter=0

#         return data
    
#     def batch_preprocessing(self, X, mask_init):
        
#         """
#         Inputs:
#             dataset to corrupt
#             % of data to eliminate[0,1]
#             rand random state
#             replace with = 'zero' or 'nan'
#         Outputs:
#             corrupted Dataset 
#             binary mask
#         """
#         X_1d = X.flatten()
#         mask_1d = torch.ones(X_1d.shape[0])#torch.ones(self.graph_size*self.dataset_init_dim)

#         if self.corrupt_size==0:
#             # Base case (not corrupt dataset)
#             return X, mask_1d.reshape(X.shape)

#         # Take only not test indices i.e. indices with 0 in mask_init
#         available_idx = (mask_init.flatten() == 1).nonzero(as_tuple=True)[0] # self.batch_flatten_idx[mask_init.flatten() == 1]
        
#         # Randomly select indices to corrupt
#         p = torch.ones(available_idx.shape)
#         p = p / p.sum()
#         temp_idx = torch.multinomial(p, num_samples=self.corrupt_size, replacement=False)
#         corrupt_idx = available_idx[temp_idx].type(torch.int64)
        
#         X_1d[corrupt_idx] = self.cfg.imputation.fill_val
#         mask_1d[corrupt_idx] = 0
        
#         X_deg = X_1d.reshape(X.shape)
#         mask_arti = mask_1d.reshape(X.shape)

#         # Map new corrupted categorical values to appropriate label, hence later it can be transformed into DenseEmb
#         X_deg = self.batch_to_dense(X_deg, mask_arti)
        
#         return X_deg, mask_arti
    
    
    

# class ValImputeDataset(ImputationBase):
#     def get(self, idx):
#         """
#         idx: batch index
#         """
    
#         torch.manual_seed(int(idx * self.cfg.val_seed))
#         np.random.seed(int(idx * self.cfg.val_seed))
#         random.seed(int(idx * self.cfg.val_seed))
        
#         idxs = self.stable_idx[idx]  
#         #
        
#         features_clean, features_val_mcar,\
#         _, label,\
#         _, mask_arti = self.extract_batch(idxs, mode="val")
        
#         # x - features that has been corrupted artificially and initially
#         # x_init - fully clean features
#         # x_clean_deg - features which were corrupted only initially
#         data = Data(x=features_val_mcar, mask=mask_arti,
#                     x_clean = features_clean,
#                     y=torch.tensor(label).type(torch.int64),    
#                     num_idx=self.num_idx, cat_idx=self.cat_idx,
#                     cat_dims=self.cat_dims, NumFillWith=self.NumFillWith,
#                     CatFillWith=self.CatFillWith, 
#                     MostFreqClass=self.MostFreqClass,
#                     unique_idx = torch.tensor(idxs).type(torch.int64), idx=idx,
#                     mode=torch.tensor([0]).type(torch.int64))

        
      
#         return data
        
#     def prepare_validation_MCAR_once(self, ):
#         torch.manual_seed(int(self.cfg.val_seed))
#         np.random.seed(int(self.cfg.val_seed))
#         random.seed(int(self.cfg.val_seed))
        
#         # We have to degrage the whole val dataset once and keep it
#         self.features_mcar, self.MASK_val = self.data_val_preprocessing(
#                                                 self.features,
#                                                 self.MASK_init
#                                                 )
        
#     def batch_preprocessing(self,  X, mask_init):
#         return self.batch_to_dense(X, mask_init)
        
#     def data_val_preprocessing(self, X, mask_init):
        
#         """
#         Inputs:
#             dataset to corrupt
#             % of data to eliminate[0,1]
#             rand random state
#             replace with = 'zero' or 'nan'
#         Outputs:
#             corrupted Dataset 
#             binary mask
#         """
#         X_1d = X.flatten()
#         mask_1d = torch.ones(X_1d.shape[0])#torch.ones(self.graph_size*self.dataset_init_dim)

#         if self.corrupt_size==0:
#             # Base case (not corrupt dataset)
#             return X, mask_1d.reshape(X.shape)

#         # Take only not test indices i.e. indices with 0 in mask_init
#         available_idx = (mask_init.flatten() == 1).nonzero(as_tuple=True)[0] # self.batch_flatten_idx[mask_init.flatten() == 1]
        
#         # Randomly select indices to corrupt
#         p = torch.ones(available_idx.shape)
#         p = p / p.sum()
#         temp_idx = torch.multinomial(p, num_samples=self.corrupt_size, replacement=False)
#         corrupt_idx = available_idx[temp_idx].type(torch.int64)
        
#         X_1d[corrupt_idx] = self.cfg.imputation.fill_val
#         mask_1d[corrupt_idx] = 0
        
#         X_deg = X_1d.reshape(X.shape)
#         mask_arti = mask_1d.reshape(X.shape) 
#         return X_deg, mask_arti

        

# class TestImputeDataset(ImputationBase):
#     def get(self, idx):
        
#         idxs = self.stable_idx[idx] 

#         features_clean, _,\
#         features_test, label,\
#         mask_init, _ = self.extract_batch(idxs, seed_idx=idx, mode="test")

#         # x - features that has been corrupted artificially and initially
#         # x_clean - fully clean features
#         # x_initCorr - features which were corrupted only initially
#         data = Data(x=features_test, mask=mask_init, 
#                     x_clean = features_clean,
#                     y=torch.tensor(label).type(torch.int64),
#                     num_idx=self.num_idx, cat_idx=self.cat_idx,
#                     cat_dims=self.cat_dims, NumFillWith=self.NumFillWith,
#                     CatFillWith=self.CatFillWith, 
#                     MostFreqClass=self.MostFreqClass, 
#                     unique_idx = torch.tensor(idxs).type(torch.int64),
#                     mode=torch.tensor([2]).type(torch.int64))


#         return data

#     def batch_preprocessing(self,  X, mask_init):
#         return self.batch_to_dense(X, mask_init)
 