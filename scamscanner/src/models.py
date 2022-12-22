import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from .conformer import ConformerEncoder, ConformerPooler
from .layers import Perceptron

from transformers import LongformerModel, LongformerTokenizer


class ScamScanner_BoW(pl.LightningModule):
    r"""Pytorch Lightning system to train a classifier for scam contracts.
    Arguments:
    -- 
    config: dotmap.DotMap
        Configuration choices loaded from a YAML file
    """
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        self.create_model(config)
        self.config = config

    def create_model(self, config):
        self.model = Perceptron(config.model.input_dim, config.model.hidden_dim)

    def forward(self, batch):
        logit = self.classifier(batch['feat'])
        return logit

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.optimizer.lr,
            weight_decay=self.config.optimizer.weight_decay,
            eps=1e-8,
        )
        return optimizer

    def training_step(self, batch, _):
        logits = self.forward(batch)
        loss = F.binary_cross_entropy_with_logits(logits, batch['label'])
        return {'loss': loss}

    def validation_step(self, batch, _):
        logits = self.forward(batch)
        loss = F.binary_cross_entropy_with_logits(logits, batch['label'])
        return {'loss': loss}
