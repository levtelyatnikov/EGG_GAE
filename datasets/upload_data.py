import os
import numpy as np
from pathlib import Path as path

PATH = os.path.join(path.cwd(), "noised_datasets")
def load_prepared_data(dataset_name, experiment_name):
    SAVE_PATH = os.path.join(PATH, experiment_name)
    data_dir = os.path.join(SAVE_PATH, dataset_name)
    data_path = os.path.join(data_dir, f'{dataset_name}.npz')
    data = np.load(data_path, allow_pickle=True)
    return data

