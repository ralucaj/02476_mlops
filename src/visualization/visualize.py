import argparse
import sys
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from torch import nn

from src import models


def visualize():
    parser = argparse.ArgumentParser(description='Visualization arguments')
    parser.add_argument('load_model_from', default='./models/model.pth')
    # add any additional argument that you want
    args = parser.parse_args(sys.argv[1:])
    print(args)

    model = models.model.MyAwesomeModel()
    model.load_state_dict(torch.load(args.load_model_from))
    data = torch.load('./data/processed/test.pt')

    embeddings = {}
    def get_activation(name):
        def hook(model, input, output):
            embeddings[name] = output.detach()

        return hook
    model.fc3.register_forward_hook(get_activation('fc3'))

    model.eval()
    log_ps = model(data['images'])
    log_ps = nn.Softmax()(log_ps)
    _, top_class = log_ps.topk(1, dim=1)
    preds = top_class.squeeze()

    projection = TSNE(n_components=2).fit_transform(embeddings['fc3'])

    # Plot those points as a scatter plot and label them based on the pred labels
    cmap = cm.get_cmap('tab20')
    fig, ax = plt.subplots(figsize=(8, 8))
    num_categories = 10
    for lab in range(num_categories):
        indices = preds == lab
        ax.scatter(projection[indices, 0], projection[indices, 1], c=np.array(cmap(lab)).reshape(1, 4), label=lab,
                   alpha=0.5)
    ax.legend(fontsize='large', markerscale=2)
    plt.savefig('./reports/figures/tsne.png')

visualize()