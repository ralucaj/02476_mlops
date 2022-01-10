import torch
from torch.utils.data import Dataset


class CorruptedMNIST(Dataset):
    """Corrupted MNIST dataset class"""

    def __init__(self, filepath):
        """
        Corrupted dataset initializer. Reads data from filepath. Expects the data to be stored as the
        output of src/data/make_dataset, as data['images'] and data['labels'].

        :param filepath: location of the data saved as a PyTorch tensor
        """
        # Indexing
        self.filepath = filepath
        self.data = torch.load(filepath)
        self.images = self.data["images"]
        self.labels = self.data["labels"]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]
