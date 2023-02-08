import pandas as pd
import numpy as np
import sklearn.datasets
from sklearn import datasets
from scipy.io.arff import loadarff 


def loader(**kwargs):
    fixed_test = False
    X_test, y_test = [], []
    print("Loading dataset " + kwargs['dataset'] + "...")
    if kwargs['dataset'] == "Covertype":  # Multi-class classification dataset
        X, y = sklearn.datasets.fetch_covtype(return_X_y=True)
        n_ = X.shape[0]
        np.random.seed(0)
        idx_sampled = np.random.choice(a = np.arange(n_), size = 25000, replace=False)
        X, y = X[idx_sampled, :], y[idx_sampled]  # only take 10000 samples from dataset

    elif kwargs['dataset'] == "Adult" or kwargs['dataset'] == "AdultCat":  # Binary classification dataset with categorical data, if you pass AdultCat, the numerical columns will be discretized.
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

    elif kwargs['dataset'] == 'default_credit_card':
        df = pd.read_excel('/home/lev/datasets/default-credit-card/data.xls').drop(columns=['Unnamed: 0'])[1:].reset_index(drop=True)
        df.fillna(0, inplace=True)
        y = df['Y'].to_numpy()
        X = df.drop(columns=['Y']).to_numpy()

    elif kwargs['dataset'] == "Turkiye":
        path = '/home/lev/imputation/EdgeGeneration/classification_data/turkiye-student-evaluation/init_dataset/turkiye-student-evaluation_generic.csv'
        df = pd.read_csv(path)
        label_column = 'class'
        features_cols = set(df.columns) - set([label_column])    
        X = df[features_cols].to_numpy()
        y = df[label_column].to_numpy()
    elif kwargs['dataset'] == 'page_blocks':
        df = pd.read_fwf('/home/lev/datasets/page_blocks/page-blocks.data', header=None)
        y = df[10].to_numpy()
        X = df.drop(columns=[10]).to_numpy()

    elif kwargs['dataset'] == 'electrical_grid_stability':
        df = pd.read_csv('/home/lev/datasets/electrical_grid_stability/Data_for_UCI_named.csv')
        y = df['stabf'].to_numpy()
        X = df.drop(columns=['stabf']).to_numpy()
    
    elif kwargs['dataset'] == 'phishing_website':
        raw_data = loadarff('/home/lev/datasets/phishing_websites/PhishingData.arff')
        df = pd.DataFrame(raw_data[0])
        y = df['Result'].to_numpy()
        X = df.drop(columns=['Result']).to_numpy()

    elif kwargs['dataset'] == 'SUSY_small':
        fixed_test = True
        data_train = pd.read_csv(f'/home/lev/datasets/SUSY_exp/SUSY_small.csv', index_col=0)
        data_test = pd.read_csv(f'/home/lev/datasets/SUSY_exp/SUSY_test.csv', index_col=0)
        y = np.array(data_train['clase'])
        X = np.array(data_train.drop(columns=['clase']))

        y_test = np.array(data_test['clase'])
        X_test = np.array(data_test.drop(columns=['clase']))

    elif kwargs['dataset'] == 'SUSY_medium':
        fixed_test = True
        data_train = pd.read_csv(f'/home/lev/datasets/SUSY_exp/SUSY_medium.csv', index_col=0)
        data_test = pd.read_csv(f'/home/lev/datasets/SUSY_exp/SUSY_test.csv', index_col=0)
        y = np.array(data_train['clase'])
        X = np.array(data_train.drop(columns=['clase']))

        y_test = np.array(data_test['clase'])
        X_test = np.array(data_test.drop(columns=['clase']))

    elif kwargs['dataset'] == 'SUSY_big':
        fixed_test = True
        data_train = pd.read_csv(f'/home/lev/datasets/SUSY_exp/SUSY_big.csv', index_col=0)
        data_test = pd.read_csv(f'/home/lev/datasets/SUSY_exp/SUSY_test.csv', index_col=0)
        y = np.array(data_train['clase'])
        X = np.array(data_train.drop(columns=['clase']))

        y_test = np.array(data_test['clase'])
        X_test = np.array(data_test.drop(columns=['clase']))

    elif kwargs['dataset'] == 'SUSY_full':
        fixed_test = True
        data = pd.read_table('/home/lev/datasets/classification_data/SUSY/SUSY.csv', index_col=0)
        n_points = data.shape[0]
        test_size_full = 500000

        train_idx = np.arange(n_points-test_size_full, n_points)
        test_idx = np.arange(n_points-test_size_full, n_points)

        y = np.array(data_train.iloc[train_idx]['clase'])
        X = np.array(data_train.iloc[train_idx].drop(columns=['clase']))

        y_test = np.array(data_test.iloc[test_idx]['clase'])
        X_test = np.array(data_test.iloc[test_idx].drop(columns=['clase']))

    elif kwargs['dataset'] == 'abalone':
        data = pd.read_table('/home/lev/datasets/classification_data/abalone/abalone_R.dat', index_col=0)
        y = np.array(data['clase'])
        X = np.array(data.drop(columns=['clase']))

    elif kwargs['dataset'] == 'CalHousing':
        data = datasets.fetch_california_housing()
        X = data['data']
        y = data['target']

    elif kwargs['dataset'] == 'red_wine':
        data = pd.read_table('/home/lev/datasets/classification_data/wine-quality-red/wine-quality-red_R.dat', index_col=0)
        y = np.array(data['clase'])
        X = np.array(data.drop(columns=['clase']))

    elif kwargs['dataset'] == 'white_wine':
        data = pd.read_table('/home/lev/datasets/classification_data/wine-quality-white/wine-quality-white_R.dat', index_col=0)
        y = np.array(data['clase'])
        X = np.array(data.drop(columns=['clase']))
    
    elif kwargs['dataset'] == 'anuran':
        df = pd.read_csv('/home/lev/datasets/anuran/Frogs_MFCCs.csv').rename(columns={'Species':'class'})
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

    elif kwargs['dataset'] == 'eeg':
        path = '/home/lev/datasets/egg/eeg-eye-state.csv'
        df = pd.read_csv(path)
        y = df['Class'].to_numpy()
        df = df.drop(columns=['Class'])
        X = df.to_numpy()

    elif kwargs['dataset'] == 'wireless':
        path = '/home/lev/datasets/wireless/wifi_localization.txt'
        df = pd.read_table(path, header=None)
        
        y = df[7].to_numpy()
        df = df.drop(columns=[7])
        X = (-1) * df.to_numpy()

    elif kwargs['dataset'] == 'yeast':
        path = '/home/lev/datasets/classification_data/yeast/yeast_R.dat'
        df = pd.read_table(path, index_col=0)
        
        y = df['clase'].to_numpy()
        df = df.drop(columns=['clase'])
        X = df.to_numpy()

    elif kwargs['dataset'] == 'letter':
        path = '/home/lev/datasets/letter/letter.csv'
        df = pd.read_csv(path)
        y = df['class'].to_numpy()
        df = df.drop(columns=['class'])
        X = df.to_numpy()
    elif kwargs['dataset'] == 'car':
        path='/home/lev/datasets/car/car.data'
        df = pd.read_csv(path, header=None)

        for col in list(df.columns):
            unique = sorted(df[col].unique())
            mapper = {unique[i]:i for i in range(len(unique))}
            df[col] = df[col].apply(lambda x: mapper[x])

        y = df[6].to_numpy()
        df = df.drop(columns=[6])
        X = df.to_numpy()

    elif kwargs['dataset'] == 'chess':
        path = '/home/lev/datasets/chess/krkopt.data'
        df = pd.read_csv(path, header=None)
        for col in list(df.columns):
            unique = sorted(df[col].unique())
            mapper = {unique[i]:i for i in range(len(unique))}
            df[col] = df[col].apply(lambda x: mapper[x])

        y = df[6].to_numpy()
        df = df.drop(columns=[6])
        X = df.to_numpy()

    elif kwargs['dataset'] == 'connect':
        df = pd.read_csv('/home/lev/datasets/connect/connect.csv', header=None)
        for col in list(df.columns):
            unique = sorted(df[col].unique())
            mapper = {unique[i]:i for i in range(len(unique))}
            df[col] = df[col].apply(lambda x: mapper[x])

        y = df[42].to_numpy()
        df = df.drop(columns=[42])
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

            "SUSY_medium": { 'dataset':"SUSY_medium",
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

            "SUSY_big": { 'dataset':"SUSY_big",
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
                        
            "SUSY_full": { 'dataset':"SUSY_full",
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

            "page_blocks": { 'dataset':"page_blocks",
                        'scale': True,
                        'target_encode': True,
                        'one_hot_encode': False,
                        'num_classes': 5,  # for classification
                        'num_features': 11,
                        'cat_idx': [],
                        # cat_dims: will be automatically set.
                        'cat_dims': [],
                        'objective':'classification',
                        'leave_only_numerical': False
                        },
            "electrical_grid_stability": { 'dataset':"electrical_grid_stability",
                        'scale': True,
                        'target_encode': True,
                        'one_hot_encode': False,
                        'num_classes': 2,  # for classification
                        'num_features': 13,
                        'cat_idx': [],
                        # cat_dims: will be automatically set.
                        'cat_dims': [],
                        'objective':'classification',
                        'leave_only_numerical': False
                        },
            "eeg": { 'dataset':"eeg",
                        'scale': True,
                        'target_encode': True,
                        'one_hot_encode': False,
                        'num_classes': 2,  # for classification
                        'num_features': 14,
                        'cat_idx': [],
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
            "wireless": { 'dataset':"wireless",
                        'scale': True,
                        'target_encode': True,
                        'one_hot_encode': False,
                        'num_classes': 4,  # for classification
                        'num_features': 7,
                        'cat_idx': [],
                        # cat_dims: will be automatically set.
                        'cat_dims': [],
                        'objective':'classification',
                        'leave_only_numerical': False
                        },

            "CalHousing": { 'dataset':"CalHousing",
                        'scale': True,
                        'target_encode': True,
                        'one_hot_encode': False,
                        'num_classes': 1,  # for classification
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
                        
            "white_wine": { 'dataset':"white_wine",
                        'scale': True,
                        'target_encode': True,
                        'one_hot_encode': False,
                        'num_classes': 7,  # for classification
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
            "_Adult": { 'dataset':"Adult",
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
                        },
            "default_credit_card":{ 'dataset':"default_credit_card",
                        'scale': True,
                        'target_encode': True,
                        'one_hot_encode': False,
                        'num_classes': 2,  # for classification
                        'num_features': 24,
                        'cat_idx': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                        'objective':'classification',
                        'leave_only_numerical': False
                        },         
            "anuran":{ 'dataset':"anuran",
                        'scale': True,
                        'target_encode': True,
                        'one_hot_encode': False,
                        'num_classes': 10,  # for classification
                        'num_features': 25,
                        'cat_idx': [22, 23, 24],
                        'objective':'classification',
                        'leave_only_numerical': False
                        },

            # Categorical datasets
            # ["wireless_cat", "letter", "car", "chess", "connect"]
            "wireless_cat": { 'dataset':"wireless_cat",
                        'scale': True,
                        'target_encode': False,
                        'one_hot_encode': False,
                        'num_classes': 4,  # for classification
                        'num_features': 7,
                        'cat_idx': [0, 1, 2, 3, 4, 5, 6],
                        # cat_dims: will be automatically set.
                        'cat_dims': [],
                        'objective':'classification',
                        'leave_only_numerical': False
                        },
            "letter": { 'dataset':"letter",
                        'scale': False,
                        'target_encode': True,
                        'one_hot_encode': False,
                        'num_classes': 26,  # for classification
                        'num_features': 16,
                        'cat_idx': [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15],
                        # cat_dims: will be automatically set.
                        'cat_dims': [],
                        'objective':'classification',
                        'leave_only_numerical': False
                        },
             "car": { 'dataset':"car",
                        'scale': False,
                        'target_encode': True,
                        'one_hot_encode': False,
                        'num_classes': 4,  # for classification
                        'num_features': 6,
                        'cat_idx': [ 0,  1,  2,  3,  4,  5],
                        # cat_dims: will be automatically set.
                        'cat_dims': [],
                        'objective':'classification',
                        'leave_only_numerical': False
                        },

            "chess": { 'dataset':"chess",
                        'scale': False,
                        'target_encode': True,
                        'one_hot_encode': False,
                        'num_classes': 18,  # for classification
                        'num_features': 6,
                        'cat_idx': [0,  1,  2,  3,  4,  5],
                        # cat_dims: will be automatically set.
                        'cat_dims': [],
                        'objective':'classification',
                        'leave_only_numerical': False
                        },
            "connect": { 'dataset':"connect",
                        'scale': False,
                        'target_encode': True,
                        'one_hot_encode': False,
                        'num_classes': 3,  # for classification
                        'num_features': 42,
                        'cat_idx': [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
                                    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
                                    34, 35, 36, 37, 38, 39, 40, 41],
                        # cat_dims: will be automatically set.
                        'cat_dims': [],
                        'objective':'classification',
                        'leave_only_numerical': False
                        },
            
            
            "_HIGGS": { 'dataset':"HIGGS",
                        'scale': True,
                        'target_encode': False,
                        'one_hot_encode': False,
                        'num_classes': 1,  # should be 1 for binary classification
                        'num_features': 28,
                        'cat_idx': [27],
                        'objective':'classification',
                        'leave_only_numerical': False
                        },

            "_Covertype": {'dataset':"Covertype",
                            'scale': True,
                            'target_encode': True, 
                            'num_classes': 7,
                            'num_features': 54,
                            'objective':'classification',
                            'cat_idx':[10, 12, 42],
                            'one_hot_encode': False,
                            'leave_only_numerical': False
                            },
          
            "_Turkiye": {'dataset':"Turkiye",
                        'scale': True,
                        'target_encode': True, 
                        'num_classes': 0,
                        'num_features': 32,
                        'objective':'classification',
                        'cat_idx':[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
                        'one_hot_encode': False,
                        'leave_only_numerical': False}}    
        
    print(args[dataset_name])
    features, labels, fixed_test = loader(**args[dataset_name])
    # features, labels = np.array(features).astype(np.float64), np.array(labels)
    cat_idx = args[dataset_name]['cat_idx']
        
    return features, labels, cat_idx, fixed_test


