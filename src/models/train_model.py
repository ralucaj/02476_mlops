import matplotlib.pyplot as plt
import numpy as np
import torch
from model import MyAwesomeModel
from torch import nn, optim


def train(model_name="model"):
    print("Training day and night")
    model = MyAwesomeModel()
    train_set = torch.load("./data/processed/train.pt")

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    epochs = 30

    train_losses = []
    for e in range(epochs):
        model.train()
        running_loss = 0
        batch_size = 100
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

        print(f"Epoch {e}: Training loss {np.mean(train_losses)}\n")

    # Save training
    plt.plot(range(len(train_losses)), train_losses)
    plt.title("Training loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("./reports/figures/training_loss.png")

    # Save model
    torch.save(model.state_dict(), f"./models/{model_name}.pth")


train()
