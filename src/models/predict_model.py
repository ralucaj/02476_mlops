import argparse
import sys

import torch
from torch import nn

from model import MyAwesomeModel


def predict():
    parser = argparse.ArgumentParser(description='Prediction arguments')
    parser.add_argument('load_model_from', default='./models/model.pth')
    parser.add_argument('test_set', default='./models/model.pth')
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

    top_class.save('predictions.pt')
