import pickle
import os


def save_obj(file_path, obj):
    with open(file_path, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_obj(file_path):
    with open(file_path, 'rb') as handle:
        b = pickle.load(handle)


def get_result_filepath(results_name, root='', subfolder=''):

    res_path = os.path.join(root, 'results')
    if not os.path.isdir(res_path):
        os.mkdir(res_path)

    subfolder_path = os.path.join(res_path, subfolder)
    if not os.path.isdir(subfolder_path):
        os.mkdir(subfolder_path)

    taken = True
    index = 0
    filepath = None

    while taken:
        filepath = os.path.join(subfolder_path, 'r' + str(index) + '_' + results_name)
        if os.path.isfile(filepath):
            index += 1
        else:
            taken = False

    return filepath



