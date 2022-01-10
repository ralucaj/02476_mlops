from src.models.model import MyAwesomeModel
import numpy as np
from hydra import compose, initialize
import pytest
import torch


def test_model():
    with initialize(config_path="."):
        cfg = compose(config_name="config.yaml")
        model = MyAwesomeModel(cfg.model)
        x = torch.Tensor(np.zeros((1, 28, 28)))
        y = model(x)
        assert y.shape == (1, 10)


def test_error_on_wrong_shape():
    with pytest.raises(ValueError, match="Expected input to a 3D tensor"), initialize(
        config_path="."
    ):
        cfg = compose(config_name="config.yaml")
        model = MyAwesomeModel(cfg.model)
        model(torch.randn(1, 2, 3, 4))
