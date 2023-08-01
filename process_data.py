
import numpy as np
import os


data_dir = './datasets'
dataset = 'seeds'
data = np.load(os.path.join(data_dir,  '{}.npy'.format(dataset)), allow_pickle=True).item()

print(data['X_true'])

np.save(os.path.join(data_dir, '{}_complete.npy'.format(dataset)), data['X_true'])