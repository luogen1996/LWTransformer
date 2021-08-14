import pickle
import numpy as np
def save(data, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)

def load(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f,encoding='iso-8859-1')

def complete_data(data, batchsize):
    if data.shape[0] == batchsize:
        return data
    if len(data.shape) == 1:
        t_data = np.zeros(batchsize - data.shape[0])
        return np.hstack((data, t_data))
    else:
        shape = (batchsize - data.shape[0],) + data.shape[1:]
        t_data = np.zeros(shape)
        return np.vstack((data, t_data))
