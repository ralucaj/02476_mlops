import torch
import numpy as np

def mnist():
    # exchange with the corrupted mnist dataset
    filepath = './data/corruptmnist'
    data_files = [f'{filepath}/train_{idx}.npz' for idx in range(5)]
    train = {'images': [], 'labels': []}

    for data_file in data_files:
        with np.load(data_file) as data:
            train['images'].append(data['images'])
            train['labels'].append(data['labels'])
    test = np.load(f'{filepath}/test.npz')

    train['images'] = np.concatenate(train['images'], axis=0)
    train['labels'] = np.concatenate(train['labels'], axis=0)


    # Turn to dict and remove unwanted keys
    train = {
        'images': torch.tensor(train['images']).float(),
        'labels': torch.tensor([train['labels']]).squeeze()
    }

    test = {
        'images': torch.tensor(test['images']).float(),
        'labels': torch.tensor(test['labels']).squeeze()
    }

    return train, test
