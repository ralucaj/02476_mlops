import torch.nn.functional as F
from torch import nn, optim
from pytorch_lightning import LightningModule


class MyAwesomeModel(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg.x_dim, cfg.hidden_dim)
        self.fc2 = nn.Linear(cfg.hidden_dim, cfg.hidden_dim_2)
        self.fc3 = nn.Linear(cfg.hidden_dim_2, cfg.hidden_dim_3)
        self.fc4 = nn.Linear(cfg.hidden_dim_3, cfg.output_dim)

        # Dropout module with 0.2 drop probability
        self.dropout = nn.Dropout(p=cfg.dropout)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=cfg.lr)

    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)

        # Now with dropout
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))

        # output so no dropout here
        x = F.log_softmax(self.fc4(x), dim=1)

        return x

    def training_step(self, batch, batch_idx):
        data, target = batch
        preds = self.forward(data)
        loss = self.criterion(preds, target)
        self.log('loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        preds = self.forward(data)
        loss = self.criterion(preds, target)
        self.log('valid_loss', loss)
        return loss

    def configure_optimizers(self):
        return self.optimizer
