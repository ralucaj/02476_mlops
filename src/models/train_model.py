import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from model import MyAwesomeModel
from torch import nn, optim
import hydra
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import logging
log = logging.getLogger(__name__)
from src.data.mnist import CorruptedMNIST


@hydra.main(config_path="configs", config_name="mnist_config.yaml")
def train(cfg):
    print("Training day and night")
    model = MyAwesomeModel(cfg.model)


    train_loader = DataLoader(CorruptedMNIST(cfg.training.train_set), batch_size=cfg.training.batch_size)
    validation_loader = DataLoader(CorruptedMNIST(cfg.training.valid_set), batch_size=cfg.training.batch_size)

    early_stopping_callback = EarlyStopping(
        monitor="valid_loss", patience=3, verbose=True, mode="min"
    )
    trainer = Trainer(
        max_epochs=cfg.training.epochs,
        accelerator='gpu',
        gpus=1,
        limit_train_batches=cfg.training.limit_train_batches,
        callbacks=[early_stopping_callback]
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=validation_loader)

    # Save model
    torch.save(model.state_dict(), cfg.training.model_path)


train()
