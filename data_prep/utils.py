import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from .miss_utils import introduce_missingness



def missingness_pipeline(X, 
        num_cols:list,
        cat_cols: list,
        cat_dims: list,

        p_miss: float,
        miss_algo: str,
        miss_seed: int,

        def_fill_val: int,
        caculate_statistics=True,
        **args
        ):

    """
    Inputs:
        X: clean dataset to corrupt
        missingness: % of data to eliminate[0,1]
        
        default_fill_val: replace with special value
        seed: rand random state
        
        Outputs:
        X_degrated: corrupted Dataset
        mask: binary mask

    """
    
    # mask has value 1 at position (i,j) 
    # if value is not corrupted.
    X_degrated, mask = introduce_missingness(
        X=X.copy(), 
        p_miss=p_miss, 
        mecha=miss_algo,
        def_fill_val=def_fill_val, 
        seed=miss_seed, 
        ) 

    assert X_degrated.shape == X.shape and X.shape == mask.shape

    """
    The data is corrupted. 
    X_degrated consist of "default_fill_val" 
    if val is corrupted and original value 
    if val is not corrupted. 
    """
    # In case we want to intoduce missingness
    # into the test set and fill with 
    # statistics from the train set
    if caculate_statistics == True:
        # Now we can calculate statistics of the data.
        median_num, most_freq_class_cat =\
                calculate_statistics(X_degrated, mask, num_cols, cat_cols)

        # Prepare mappers for categorical columns
        cat_mappers, cat_mappers_freq, CatFillWith =\
            prepare_cat_mappers(cat_cols, cat_dims, most_freq_class_cat, def_fill_val)
    else:
        # Variables has to be passed to the function
        median_num = args['median_num']
        most_freq_class_cat = args['most_freq_class_cat']
        cat_mappers = args['cat_mappers']
        cat_mappers_freq = args['cat_mappers_freq']
        CatFillWith = args['CatFillWith']

    
    # Fill corrupted data with statistics
    X_deg = fill_data(data=X_degrated.copy(), mask=mask, num_cols=num_cols, cat_cols=cat_cols, NumFillWith=median_num, cat_mappers=cat_mappers)

    X_deg_freq = fill_data(data=X_degrated.copy(), mask=mask, num_cols=num_cols, cat_cols=cat_cols, NumFillWith=median_num, cat_mappers=cat_mappers_freq)
    
    if len(cat_cols)==0:
        assert (X_deg_freq == X_deg).all(), 'Should be equeal if there is not categorical columns'
    if len(num_cols)>0:
        assert (X_deg_freq[:, num_cols] == X_deg[:, num_cols]).all(), 'Should be equeal'

    return X_deg, X_deg_freq, mask, median_num, CatFillWith, most_freq_class_cat, cat_mappers, cat_mappers_freq

def fill_data(data, mask, num_cols, cat_cols, NumFillWith, cat_mappers):
    
    """

    data - corrupted data
    mask - binary mask where 1 is not corrupted and 0 is corrupted
    num_cols - numerical columns
    cat_cols - categorical columns
    NumFillWith - list of numerical values to fill with    
    """
    
    filled_data = data.copy()
    if len(cat_cols) > 0:
        assert len(num_cols) == len(NumFillWith), f"len of num_cols={len(num_cols)} doesn't coinside with len of NumFillWith={len(NumFillWith)}"
        """
        
        HUGE NOTE: 
        The thing is that I use nn.Embedding in during the training to map 
        the dim value (the value which come from CatFillWith) into the most frequent class
        Hence nn.Embedding maps the dim value to the most frequent class
        It have to be done always for training, validation and test
        
        However other algos have to be directly substituted with most frequent class
        """

        # Map missed categorical categorical values with CatFillWith value
        for temp in list(zip(cat_cols, cat_mappers)):
            col, col_mapper = temp

            corrupt_idx = np.where(mask[:, col]==0)[0]

            # This operation will give categorical column
            # with ubique values as [0,1,.., specia_token=dim]
            # The *specia_token=dim* value is an indicator
            # that this value has is missed.

            filled_data[corrupt_idx, col] = np.array(list(map(lambda x: int(col_mapper[x]), filled_data[corrupt_idx, col])))
        
    if len(num_cols) > 0:
        # Fill with meadian of numerical col
        for col, median_fill_with in zip(num_cols, NumFillWith):
            corrupt_idx = np.where(mask[:, col]==0)[0]

            filled_data[corrupt_idx, col] = median_fill_with
    
    return filled_data
    
def calculate_statistics(data, mask, num_cols, cat_cols):
    """
    mask (binary) - 1 where data is not corrupted and 0 where data is corrupted
    num_cols - numerical columns
    cat_cols - categorical columns
    cat_dims - categorical dimensions (each cat column has its number of classes)
    default_fill_val - value to indicate that data is corrupted inside data (only for categorical values)
    directly_fill_mostfreq_class - if True then most frequent class will be directly substituted in the data
    """
    assert data.shape[1] == (len(num_cols) + len(cat_cols)),\
        f"{data.shape[1]} doesn't coinside with num_cols + cat_cols = {len(num_cols) + len(cat_cols)}"
    
    median_num, most_freq_class_cat = [], []
    
    
    if len(cat_cols) > 0:
        for col in cat_cols:
            # Extract values which are observered
            observer_idx = np.where(mask[:, col]==1)[0]
            l = list(data[observer_idx, col])
            
            # Find most frequent class
            most_freq_class_cat.append(max(set(l), key=l.count))
    
    
    if len(num_cols) > 0:
        for col in num_cols:
            # Extract values which are observered
            observer_idx = np.where(mask[:, col]==1)[0]

            # Extract meadian of numerical col
            median = np.nanmedian(data[observer_idx, col])
            median_num.append(median)
    

    return  median_num, most_freq_class_cat
    
def scaler(X_train, X_val, num_idx, X_clean=None, scale_type='StandardScaler'):
    flag = 0
    if len(num_idx) > 0:
        if scale_type == 'StandardScaler':
            scaler_f = StandardScaler()
            flag = 1
        elif scale_type=='MinMaxScaler':
            scaler_f = MinMaxScaler((0, 1))
            flag = 1
        elif scale_type == 'None':
            scaler_f = None
            flag = 0 
    else:
        scaler_f = None
        

    if flag == 1:
        X_train[:, num_idx] = scaler_f.fit_transform(X_train[:, num_idx])
        X_val[:, num_idx] = scaler_f.transform(X_val[:, num_idx])
        if X_clean is not None:
            X_clean[:, num_idx] = scaler_f.transform(X_clean[:, num_idx])

    
    return X_train, X_val, X_clean, scaler_f
   

def prepare_cat_mappers(cat_cols, cat_dims, most_freq_class_cat, default_fill_val):
    assert len(cat_cols)==len(cat_dims), f"len of cat_cols={len(cat_cols)} doesn't coinside with len of cat_dims={len(cat_dims)}" 
    assert len(cat_cols)==len(most_freq_class_cat), f"len of cat_cols={len(cat_cols)} doesn't coinside with len of most_freq_class_cat={len(most_freq_class_cat)}"
    
    cat_mappers, cat_mappers_freq = [], []
    CatFillWith = cat_dims.copy()
    
    if len(cat_cols) > 0:
    
        for dim in CatFillWith:
            # Known classes
            temp_mapper = dict(zip(np.arange(dim),  np.arange(dim)))

            # Add default_fill_val into the dict 
            temp_mapper[default_fill_val] = dim

            # ???????????
            # temp_mapper[dim] = dim
            # ???????????
            cat_mappers.append(temp_mapper)
        
        for dim, most_freq_class in zip(cat_dims, most_freq_class_cat):
            # Known classes
            temp_mapper = dict(zip(np.arange(dim),  np.arange(dim)))

            # Add default_fill_val into the dict 
            temp_mapper[default_fill_val] = most_freq_class

            # ???????????
            # temp_mapper[dim] = dim
            # ???????????
            
            cat_mappers_freq.append(temp_mapper)
    return cat_mappers, cat_mappers_freq, CatFillWith
    
    
   

