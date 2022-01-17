import logging

import hydra
import torch
from model import MyAwesomeModel
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader

from src.data.mnist import CorruptedMNIST

log = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="mnist_config.yaml")
def train(cfg):
    print("Training day and night")
    model = MyAwesomeModel(cfg.model)

    train_loader = DataLoader(
        CorruptedMNIST(cfg.training.train_set), batch_size=cfg.training.batch_size
    )
    validation_loader = DataLoader(
        CorruptedMNIST(cfg.training.valid_set), batch_size=cfg.training.batch_size
    )

    early_stopping_callback = EarlyStopping(
        monitor="valid_loss", patience=3, verbose=True, mode="min"
    )
    trainer = Trainer(
        max_epochs=cfg.training.epochs,
        accelerator="gpu",
        gpus=1,
        limit_train_batches=cfg.training.limit_train_batches,
        callbacks=[early_stopping_callback],
    )
    trainer.fit(
        model, train_dataloaders=train_loader, val_dataloaders=validation_loader
    )

    # Save model
    torch.save(model.state_dict(), cfg.training.model_path)

    script_model = torch.jit.script(model)
    script_model.save('deployable_model.pt')


train()
