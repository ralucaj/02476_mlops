import argparse
import sys

import numpy as np
import torch
from model import MyAwesomeModel
from torch import nn, optim

from data import mnist


class TrainOREvaluate(object):
    """Helper class that will help launch class methods as commands
    from a single script
    """

    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>",
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print("Unrecognized command")

            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()

    def train(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description="Training arguments")
        parser.add_argument("--lr", default=0.1)
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)

        # TODO: Implement training loop here
        model = MyAwesomeModel()
        train_set, test_set = mnist()

        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.003)

        epochs = 30
        steps = 0

        train_losses = []
        for e in range(epochs):
            model.train()
            running_loss = 0
            batch_size = 100
            for batch in range(len(train_set["images"]) // batch_size):
                optimizer.zero_grad()

                log_ps = model(
                    train_set["images"][batch * batch_size : (batch + 1) * batch_size]
                )
                loss = criterion(
                    log_ps,
                    train_set["labels"][batch * batch_size : (batch + 1) * batch_size],
                )
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            train_losses.append(running_loss)

            print(f"Epoch {e}: Training loss {np.mean(train_losses)}\n")
        torch.save(model.state_dict(), "model.pth")

    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description="Training arguments")
        parser.add_argument("load_model_from", default="./model.pth")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)

        # TODO: Implement evaluation logic here
        model = MyAwesomeModel()
        model.load_state_dict(torch.load(args.load_model_from))

        _, test_set = mnist()

        criterion = nn.NLLLoss()

        test_losses = []
        preds = torch.Tensor([])
        labels = torch.Tensor([])
        model.eval()
        batch_size = 100
        with torch.no_grad():
            running_loss = 0
            for batch in range(len(test_set["images"]) // batch_size):
                log_ps = model(
                    test_set["images"][batch * batch_size : (batch + 1) * batch_size]
                )
                loss = criterion(
                    log_ps,
                    test_set["labels"][batch * batch_size : (batch + 1) * batch_size],
                )
                running_loss += loss.item()
                log_ps = nn.Softmax()(log_ps)
                _, top_class = log_ps.topk(1, dim=1)
                top_class = top_class.squeeze()
                preds = torch.cat((preds, top_class))
                labels = torch.cat(
                    (
                        labels,
                        test_set["labels"][
                            batch * batch_size : (batch + 1) * batch_size
                        ],
                    )
                )

        test_losses.append(running_loss)

        equals = preds == labels
        accuracy = torch.mean(equals.type(torch.FloatTensor))
        print(f"Test loss {np.mean(test_losses)}, test accuracy {accuracy * 100}%\n")


if __name__ == "__main__":
    TrainOREvaluate()
