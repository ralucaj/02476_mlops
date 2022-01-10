from src.data.mnist import CorruptedMNIST
import numpy as np
import pytest
import os


@pytest.mark.skipif(not os.path.exists("data/processed/train.pt"), reason="Data files not found")
def test_corrupted_mnist():
    data = CorruptedMNIST("data/processed/train.pt")
    assert len(data) == 40000, "Dataset did not have the correct number of samples"

    test_data = CorruptedMNIST("data/processed/test.pt")
    assert len(test_data) == 5000, "Dataset did not have the correct number of samples"

    labels = []
    for idx in range(len(data)):
        data_item = data[idx]
        assert (28, 28) == data_item[0].shape
        labels.append(data_item[1])

    assert np.array_equal(np.unique(labels), np.array(range(10)))
