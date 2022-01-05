import matplotlib.pyplot as plt
import numpy as np
import torch
from model import MyAwesomeModel
from torch import nn, optim
import hydra
import logging
log = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="mnist_config.yaml")
def train(cfg):
    print("Training day and night")
    model = MyAwesomeModel(cfg.model)
    train_set = torch.load(cfg.training.train_set)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.training.lr)

    epochs = cfg.training.epochs

    train_losses = []
    for e in range(epochs):
        model.train()
        running_loss = 0
        batch_size = cfg.training.batch_size
        for batch in range(len(train_set["images"]) // batch_size):
            optimizer.zero_grad()

            log_ps = model(
                train_set["images"][batch * batch_size:(batch + 1) * batch_size]
            )
            loss = criterion(
                log_ps,
                train_set["labels"][batch * batch_size:(batch + 1) * batch_size],
            )
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        train_losses.append(running_loss)

        logging.info(f"Epoch {e}: Training loss {np.mean(train_losses)}\n")

    # Save training
    plt.plot(range(len(train_losses)), train_losses)
    plt.title("Training loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(cfg.training.figures_path)

    # Save model
    torch.save(model.state_dict(), cfg.training.model_path)


train()
