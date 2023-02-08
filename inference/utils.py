import os
import shutil
def cat_dicts(a, b):
    return dict(list(a.items()) + list(b.items()))

def cat_dicts3(a, b, c):
    return dict(list(a.items()) + list(b.items() )+ list(c.items()))

def delete_folder(dir_path):
    if os.path.isdir(dir_path):
        #print('Deleting folder')
        shutil.rmtree(dir_path)