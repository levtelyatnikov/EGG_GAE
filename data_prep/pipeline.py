from .upload import upload_data
from .preprocessing import preprocess_data
from .save import save_data

def data_pipeline(dataset_name, experiment_name,
    def_fill_val,
    p_miss,
    miss_algo,
    miss_seed, 
    val_size,
    data_split_seed,
    scale_type):
    
    # Upload data
    features, labels, cat_cols, fixed_test = upload_data(dataset_name)
    

    # Preprocess
    X_train, X_val, X_test, X_train_clean, X_val_clean, X_test_clean,\
    y_train, y_val, y_test,\
    mask_train, mask_val, mask_test,\
    X_train_freq, X_val_freq, X_test_freq,\
    NumFillWith, CatFillWith, \
    MostFreqClass, cat_mappers, \
    num_idx, cat_idx, cat_dims = preprocess_data(X=features, 
                                                y=labels,
                                                cat_cols=cat_cols,
                                                def_fill_val=def_fill_val,
                                                p_miss=p_miss,
                                                miss_algo=miss_algo,
                                                miss_seed=miss_seed, 
                                                val_size=val_size,
                                                data_split_seed=data_split_seed,
                                                scale_type=scale_type,
                                                fixed_test=fixed_test)
        
    

    # Save prepared dataset
    assert (y_val == y_test).all()
    #assert (mask_val = mask_test).all()
    
    # Mask_train and mask_test have to be the same
    # During train I will use mask_train when introducing additional missing values
    # such that the values which are for the test set are not participating in the training
    # so avoid data leakage
    save_data(X_train_clean=X_train_clean, X_val_clean=X_val_clean, X_test_clean=X_test_clean,\
                y_train=y_train, y_val=y_val, y_test=y_test, \
                X_train_deg=X_train, mask_init_train=mask_train, \
                X_val_deg=X_val, mask_init_val=mask_val, \
                X_test_deg=X_test, mask_init_test=mask_test, \
                NumFillWith=NumFillWith, CatFillWith=CatFillWith, \
                MostFreqClass=MostFreqClass, cat_mappers=cat_mappers, \
                num_idx=num_idx, cat_idx=cat_idx, dataset_name=dataset_name, cat_dims=cat_dims, \
                X_train_deg_freq = X_train_freq, X_val_deg_freq = X_val_freq, X_test_deg_freq = X_test_freq,
                experiment_name=experiment_name)

