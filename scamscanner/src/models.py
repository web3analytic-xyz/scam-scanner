import torch
import torch.nn.functional as F

import pytorch_lightning as pl
from .layers import ResPerceptron
from .utils import collect_metrics


class ScamScanner(pl.LightningModule):
    r"""Pytorch Lightning system to train a classifier for scam contracts.
    Notes:
    --
    MLP on top of TF-IDF features on contract OPCODES.
    Arguments:
    -- 
    config: dotmap.DotMap
        Configuration choices loaded from a YAML file
    """
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = ResPerceptron(config.model.input_dim)
        self.config = config

    def forward(self, batch):
        logit = self.model(batch['feat'])
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
        loss = F.binary_cross_entropy_with_logits(logits.squeeze(1), batch['label'].float())
        return {'loss': loss}

    def validation_step(self, batch, _):
        logits = self.forward(batch).squeeze(1)
        labels = batch['label'].float()
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        pred = torch.round(torch.sigmoid(logits))
        acc = torch.sum(pred == labels).item() / float(len(labels))
        return {'loss': loss, 'acc': acc}

    def test_step(self, batch, _):
        logits = self.forward(batch).squeeze(1)
        labels = batch['label'].float()
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        pred = torch.round(torch.sigmoid(logits))
        acc = torch.sum(pred == labels).item() / float(len(labels))
        return {'loss': loss, 'acc': acc}

    def training_epoch_end(self, outputs):
        metrics = collect_metrics(outputs, 'train')
        self.log_dict(metrics)

    def validation_epoch_end(self, outputs):
        metrics = collect_metrics(outputs, 'dev')
        self.log_dict(metrics)

    def test_epoch_end(self, outputs):
        metrics = collect_metrics(outputs, 'test')
        self.log_dict(metrics)
