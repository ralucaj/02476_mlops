import argparse
import sys

import torch
from model import MyAwesomeModel
from torch import nn


def predict():
    """
    Predict the class probabilities for the given data. Saves them in `predictions.pt`.

    :param load_model_from: PyTorch model filepath from which the model will be read
    :param test_set: PyTorch tensor containing the expected test images
    :return:
    """
    parser = argparse.ArgumentParser(description="Prediction arguments")
    parser.add_argument("load_model_from", default="./models/model.pth")
    parser.add_argument("test_set", default="./data/processed/train.pt")
    # add any additional argument that you want
    args = parser.parse_args(sys.argv[2:])
    print(args)

    model = MyAwesomeModel()
    model.load_state_dict(torch.load(args.load_model_from))

    test_images = torch.load(args.test_set)

    model.eval()
    with torch.no_grad():
        log_ps = model(test_images)
        log_ps = nn.Softmax()(log_ps)
        _, top_class = log_ps.topk(1, dim=1)
        top_class = top_class.squeeze()

    top_class.save("predictions.pt")
