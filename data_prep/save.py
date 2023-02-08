import os
import shutil
import numpy as np
from pathlib import Path as path


PATH = os.path.join(path.cwd(), "noised_datasets")

def delete_folder(dataset_name, experiment_name):
    dir_path = os.path.join(PATH, experiment_name)
    dir_path = os.path.join(dir_path, dataset_name)
    if os.path.isdir(dir_path):
        print('Deleting folder')
        shutil.rmtree(dir_path)

def save_data(dataset_name, **kwargs):
    dir_path = os.path.join(PATH, kwargs['experiment_name'])
    dir_path = os.path.join(dir_path, dataset_name)
    save_path = os.path.join(dir_path, dataset_name)
    delete_folder(dataset_name, kwargs['experiment_name'])
    os.makedirs(dir_path)
    print(f'Saving dataset, name {dataset_name}')
    np.savez(file=save_path, **kwargs)
