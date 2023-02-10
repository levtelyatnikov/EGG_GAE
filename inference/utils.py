import os
import shutil

def delete_folder(dir_path):
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)


def cat_dicts(a, b):
    return dict(list(a.items()) + list(b.items()))

# def cat_dicts3(a, b, c):
#     return dict(list(a.items()) + list(b.items() )+ list(c.items()))