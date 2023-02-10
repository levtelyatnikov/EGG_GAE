import pandas as pd
import numpy as np
from scipy.io.arff import loadarff 
import os

def path_to_data(project_name='EGG_GAE'):
    """Add project to the path"""
    working_dir = os.getcwd().split('/')
    dir_path = None
    for i, folder_name in enumerate(working_dir):
        if project_name==folder_name:
            dir_path = '/'.join(working_dir[:i+1])
            
    
    if dir_path is None:
        raise ValueError('Path to data is not found!!')
    else: 
        return dir_path
    
    
def loader(**kwargs):
    fixed_test = False
    X_test, y_test = [], []
    print("Loading dataset " + kwargs['dataset'] + "...")
    dir_path = path_to_data()
    if kwargs['dataset'] == "Adult":  # Binary classification dataset with categorical data, if you pass AdultCat, the numerical columns will be discretized.
        fixed_test = True
        url_data = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
        url_data_test = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test'
        
        features = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital-status', 'occupation',
                    'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
        label = "income"
        columns = features + [label]
        df = pd.read_csv(url_data, names=columns)
        # Fill NaN with something better?
        df.fillna(0, inplace=True)
        X = df[features].to_numpy()
        y = df[label].to_numpy()

        df = pd.read_csv(url_data_test, names=columns)
        # Fill NaN with something better?
        df.dropna(inplace=True)
        X_test = df[features].to_numpy()
        y_test = df[label].to_numpy()
    
    elif kwargs['dataset'] == 'phishing_website':
        raw_data = loadarff(f'{dir_path}/data/phishing_website/PhishingData.arff')
        df = pd.DataFrame(raw_data[0])
        y = df['Result'].to_numpy()
        X = df.drop(columns=['Result']).to_numpy()
    
    elif kwargs['dataset'] == 'SUSY_small':
        # print current working directory
        import os
        print("Current working directory: {0}".format(os.getcwd()))
        fixed_test = True
        data_train = pd.read_csv(f'{dir_path}/data/susy/SUSY_small.csv', index_col=0)
        data_test = pd.read_csv(f'{dir_path}/data/susy/SUSY_test.csv', index_col=0)
        y = np.array(data_train['clase'])
        X = np.array(data_train.drop(columns=['clase']))

        y_test = np.array(data_test['clase'])
        X_test = np.array(data_test.drop(columns=['clase']))


    elif kwargs['dataset'] == 'abalone':
        data = pd.read_table(f'{dir_path}/data/abalone/abalone_R.dat', index_col=0)
        y = np.array(data['clase'])
        X = np.array(data.drop(columns=['clase']))

   
    elif kwargs['dataset'] == 'red_wine':
        data = pd.read_table(f'{dir_path}/data/red_wine/wine-quality-red_R.dat', index_col=0)
        y = np.array(data['clase'])
        X = np.array(data.drop(columns=['clase']))

    elif kwargs['dataset'] == 'white_wine':
        data = pd.read_table(f'{dir_path}/data/wine-quality-white/wine-quality-white_R.dat', index_col=0)
        y = np.array(data['clase'])
        X = np.array(data.drop(columns=['clase']))
    
    elif kwargs['dataset'] == 'anuran':
        df = pd.read_csv(f'{dir_path}/data/anuran/Frogs_MFCCs.csv').rename(columns={'Species':'class'})
        cat_cols = ['Family', 'Genus', 'RecordID']
        y = df['class']
        df = df.drop(columns=['class'])

        for col in cat_cols:
            unique = sorted(df[col].unique())
            mapper = {unique[i]:i for i in range(len(unique))}
            df[col] = df[col].apply(lambda x: mapper[x])


        unique = sorted(y.unique())
        mapper = {unique[i]:i for i in range(len(unique))}
        df['class'] = y
        df['class'] = df['class'].apply(lambda x: mapper[x])

        label_column = 'class'


        cat_df = df[cat_cols].to_numpy()
        df = df.drop(columns=cat_cols)

        # labels
        y = df[label_column].to_numpy()
        df = df.drop(columns=[label_column])

        num_df = df.to_numpy()
        # features
        X = np.concatenate([num_df, cat_df], axis=1)


    elif kwargs['dataset'] == 'yeast':
        path = f'{dir_path}/data/yeast/yeast_R.dat'
        df = pd.read_table(path, index_col=0)
        
        y = df['clase'].to_numpy()
        df = df.drop(columns=['clase'])
        X = df.to_numpy()

    else:
        raise AttributeError("Dataset \"" + kwargs['dataset'] + "\" not available")

    print(f"Dataset loaded!, Dataset shape={X.shape}")
    return X, y, [fixed_test, X_test, y_test]

def upload_data(dataset_name):
    
    args = {
            # Numerical datasets
        # ["SUSY-small", "eeg", "yeast", "wireless", "CalHousing", "red_wine", "abalone"]
            "SUSY_small": { 'dataset':"SUSY_small",
                        'scale': True,
                        'target_encode': True,
                        'one_hot_encode': False,
                        'num_classes': 2,  # for classification
                        'num_features': 18,
                        'cat_idx': [],
                        # cat_dims: will be automatically set.
                        'cat_dims': [],
                        'objective':'classification',
                        'leave_only_numerical': False
                        },

            

            "phishing_website":{ 'dataset':"phishing_website",
                        'scale': True,
                        'target_encode': True,
                        'one_hot_encode': False,
                        'num_classes': 3,  # for classification
                        'num_features': 9,
                        'cat_idx': [0,1,2,3,4,5,6,7,8],
                        # cat_dims: will be automatically set.
                        'cat_dims': [],
                        'objective':'classification',
                        'leave_only_numerical': False
                        },

            
            "yeast": { 'dataset':"yeast",
                        'scale': True,
                        'target_encode': True,
                        'one_hot_encode': False,
                        'num_classes': 10,  # for classification
                        'num_features': 8,
                        'cat_idx': [],
                        # cat_dims: will be automatically set.
                        'cat_dims': [],
                        'objective':'classification',
                        'leave_only_numerical': False
                        },
            
            "red_wine": { 'dataset':"red_wine",
                        'scale': True,
                        'target_encode': True,
                        'one_hot_encode': False,
                        'num_classes': 6,  # for classification
                        'num_features': 11,
                        'cat_idx': [],
                        # cat_dims: will be automatically set.
                        'cat_dims': [],
                        'objective':'classification',
                        'leave_only_numerical': False
                        },
                        
            

            "abalone": { 'dataset':"abalone",
                        'scale': True,
                        'target_encode': True,
                        'one_hot_encode': False,
                        'num_classes': 3,  # for classification
                        'num_features': 8,
                        'cat_idx': [],
                        # cat_dims: will be automatically set.
                        'cat_dims': [],
                        'objective':'classification',
                        'leave_only_numerical': False
                        },

            # Mixed datasets
            "Adult": { 'dataset':"Adult",
                        'scale': True,
                        'target_encode': True,
                        'one_hot_encode': False,
                        'num_classes': 2,  # for classification
                        'num_features': 14,
                        'cat_idx': [1,3,5,6,7,8,9,13],
                        # cat_dims: will be automatically set.
                        'cat_dims': [9, 16, 7, 15, 6, 5, 2, 42],
                        'objective':'classification',
                        'leave_only_numerical': False
                        }}    
        
    print(args[dataset_name])
    features, labels, fixed_test = loader(**args[dataset_name])
    # features, labels = np.array(features).astype(np.float64), np.array(labels)
    cat_idx = args[dataset_name]['cat_idx']
        
    return features, labels, cat_idx, fixed_test


