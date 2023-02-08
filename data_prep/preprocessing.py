import numpy as np
from sklearn.preprocessing import LabelEncoder
from data_prep.utils import scaler, missingness_pipeline
from sklearn.model_selection import train_test_split


def label_encoder(y):
    le = LabelEncoder()
    return le.fit_transform(y), len(le.classes_)

def prepare_cat_columns(X, y, cat_idx):
    num_features = X.shape[1]
    num_idx = []
    cat_dims = []

    # Map each col features into [0, .., n_classes-1]
    # Extract number of unique categories of column 
    for i in range(num_features):
        if cat_idx and i in cat_idx:
            col, n_unique = label_encoder(X[:, i])
            X[:, i] = col
            cat_dims.append(n_unique)
        else:
            num_idx.append(i)
    X, y = np.array(X).astype(np.float64), np.array(y)
    return X, y, num_idx, cat_dims

def isnan(X):
    return np.isnan(X).all()

def preprocess_data(X, y, cat_cols, 
    def_fill_val,
    p_miss,
    miss_algo,
    miss_seed, 
    val_size,
    data_split_seed,
    scale_type,
    fixed_test):
    """
    
    X, y numpy arrays
    cat_cols: indexes of categorical features
     
    X_deg: consists of categorical values filled with 
    categorical special token = |n_classes| + 1. 
    X_deg_freq: consists of most frequent class
    for each categorical feature.
    """

    assert X.shape[0] > 0 and  y.shape[0] > 0, 'data has not been uploaded'
    assert X.shape[1] > 0, 'data has issues with features'
    
    # Preprocess target
    y, _ = label_encoder(y)
 
    num_cols = []
    cat_dims = []

    # Preprocess data
    # Map each col features into [0, .., n_classes-1]
    # Extract number of unique categories of column 
    X, y, num_cols, cat_dims = prepare_cat_columns(X, y, cat_cols)
    
    # Check if data has errors
    if isnan(X):
        Exception('Data has NaNs')
    
    
    

    if fixed_test[0]==True:
        # Then X is the whole training set
        X_train_clean, y_train = X.copy(), y.copy()
        # Mask has 0 where data is missing
        X_deg_train, X_deg_train_freq, mask_train,\
        NumFillWith, CatFillWith,\
        MostFreqClass,\
        cat_mappers, cat_mappers_freq = missingness_pipeline(
            X=X.copy(), 
            num_cols=num_cols,
            cat_cols=cat_cols,
            cat_dims=cat_dims,

            p_miss=p_miss,
            miss_algo=miss_algo,
            miss_seed=miss_seed,

            def_fill_val=def_fill_val)

        # Some datasets can have prefixed test set
        # For example, in the case of the Adult dataset
        # To takle this we consider the test set as a validation set.
        # Note that we introduce missing values into validation
        # Hence we optimize the model with respect to artificial noise
        # When we perform testing we do not introduce
        # any additional missing values.
        
        X_test_clean, y_test = fixed_test[1:]
        
        # Preprocess target
        y_test, _ = label_encoder(y_test)

        assert set(np.unique(y_test)) == set(np.unique(y))
        # Preprocess data
        # Map each col features into [0, .., n_classes-1]
        # Extract number of unique categories of column 
        X_test_clean, y_test, _, _ = prepare_cat_columns(X_test_clean, y_test, cat_cols)
        
        # Check if data has errors
        if isnan(X_test_clean):
            Exception('Data has NaNs')

        
        X_deg_test, X_deg_test_freq, mask_test,\
        _, _, _, _, _ = missingness_pipeline(
            X=X_test_clean.copy(),

            num_cols=num_cols,
            cat_cols=cat_cols,
            cat_dims=cat_dims,
            p_miss=p_miss,
            miss_algo=miss_algo,
            miss_seed=miss_seed,

            def_fill_val=def_fill_val,

            # Takes stat from training set
            caculate_statistics=False,
            # To provided stat fron train set
            median_num = NumFillWith,
            most_freq_class_cat = MostFreqClass,
            cat_mappers = cat_mappers,
            cat_mappers_freq = cat_mappers_freq,
            CatFillWith = CatFillWith)
        
        print()

    else:
        X_train_clean, X_test_clean, y_train, y_test = train_test_split(X.copy(), y.copy(),
            test_size=val_size,    
            # Train validation can be not fixed though
            random_state=data_split_seed,
            stratify=y)


        X_deg_train, X_deg_train_freq, mask_train,\
        NumFillWith, CatFillWith,\
        MostFreqClass,\
        cat_mappers, cat_mappers_freq = missingness_pipeline(
            X=X_train_clean.copy(), 
            num_cols=num_cols,
            cat_cols=cat_cols,
            cat_dims=cat_dims,

            p_miss=p_miss,
            miss_algo=miss_algo,
            miss_seed=miss_seed,

            def_fill_val=def_fill_val)

        # Introduce initial missing values into validation set
        X_deg_test, X_deg_test_freq, mask_test,\
        _, _, _, _, _ = missingness_pipeline(
            X=X_test_clean.copy(),
            num_cols=num_cols,
            cat_cols=cat_cols,
            cat_dims=cat_dims,
            p_miss=p_miss,
            miss_algo=miss_algo,
            miss_seed=miss_seed,

            def_fill_val=def_fill_val,

            # Take stat from training set
            caculate_statistics=False,

            # To provided stat fron train set
            median_num = NumFillWith,
            most_freq_class_cat = MostFreqClass,
            cat_mappers = cat_mappers,
            cat_mappers_freq = cat_mappers_freq,
            CatFillWith = CatFillWith)
    
    # Scale numerical features of the dataset 
    # Missed numerical features are filled with median
    
    # We devided to scale X_train_clean with 
    # statistics estimated from X_deg_train
   
    X_deg_train, X_deg_test, X_train_clean, scaler_f = scaler(
        X_train=X_deg_train.copy(), 
        X_val=X_deg_test.copy(), 
        X_clean=X_train_clean.copy(),
        scale_type=scale_type, 
        num_idx=num_cols
        )
    
    # Scale the clean validation as well
    X_deg_train_freq, X_deg_test_freq, X_test_clean, _ = scaler(
        X_train=X_deg_train_freq.copy(), 
        X_val=X_deg_test_freq.copy(), 
        X_clean=X_test_clean.copy(),
        scale_type=scale_type, 
        num_idx=num_cols
        )
    

    assert (X_deg_train[:, num_cols] ==  X_deg_train_freq[:, num_cols]).all(), 'Numerical features have to be the same'

    # Make validation/test sets:
    # -------------------------------------
    # Test set: 
    # In the real-world scenario we do not know
    # the missingness mechanism. Hence we will use
    # X_test for final validation. X_test can 
    # be corrupted with MCAR, MAR or MNAR missingness.


    # X_deg_test, X_deg_test_freq = X_deg_val.copy(), X_deg_val_freq.copy()
    # X_test_clean = X_val_clean.copy()
    # X_test_clean = X_val_clean.copy()
    # y_test = y_val.copy()
    # mask_test = mask_val.copy()

    # Val set:
    # by default we always assume the validation missingness
    # mechanism is being MCAR. Hence the models
    # will be optimised on the test set filled 
    # with meadian filled values.
    
    # We assume X_test filled with estimated statistics
    # to be X_val_clean.  Additionalluy we do not know 
    # the missingness percentage of the values in the real-world
    # scenario. Hence MCAR noise introduced always with the same
    # percentage 0.2
    X_val_clean = X_deg_test_freq.copy()
    y_val = y_test.copy()

    X_deg_val, X_deg_val_freq, mask_val,\
        _, _, _, _, _ = missingness_pipeline(
            X=X_deg_test_freq.copy(), 
            miss_algo="MCAR",
            p_miss=0.2, #p_miss

            num_cols=num_cols,
            cat_cols=cat_cols,
            cat_dims=cat_dims,
            
            miss_seed=miss_seed + 1, # to avoid the same seed

            def_fill_val=def_fill_val,

            # Take stat from training set
            caculate_statistics=False,

            # To provided stat fron train set
            median_num = (NumFillWith - scaler_f.mean_)/scaler_f.scale_ if len(NumFillWith) else NumFillWith,
            most_freq_class_cat = MostFreqClass,
            cat_mappers = cat_mappers,
            cat_mappers_freq = cat_mappers_freq,
            CatFillWith = CatFillWith)
    
                                
    return X_deg_train, X_deg_val, X_deg_test, X_train_clean, X_val_clean, X_test_clean, \
           y_train, y_val, y_test,\
           mask_train, mask_val, mask_test,\
           X_deg_train_freq, X_deg_val_freq, X_deg_test_freq,\
           NumFillWith, CatFillWith, \
           MostFreqClass, cat_mappers, \
           num_cols, cat_cols, cat_dims
    
