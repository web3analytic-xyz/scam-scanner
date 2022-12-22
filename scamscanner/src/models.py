import torch
import torch.nn.functional as F

import pytorch_lightning as pl
from .layers import Perceptron
from .utils import collect_metrics, collect_binary_classification_metrics


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
        
        self.model = Perceptron(config.model.input_dim)
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
        probs = torch.sigmoid(logits)
        # Return the actual predictions so we can compute fine-grain metrics
        return {
            'loss': loss,
            'labels': labels.cpu().numpy(),
            'probs': probs.cpu().numpy(),
        }

    def test_step(self, batch, _):
        logits = self.forward(batch).squeeze(1)
        labels = batch['label'].float()
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        probs = torch.sigmoid(logits)
        # Return the actual predictions so we can compute fine-grain metrics
        return {
            'loss': loss,
            'labels': labels.cpu().numpy(),
            'probs': probs.cpu().numpy(),
        }

    def training_epoch_end(self, outputs):
        metrics = collect_metrics(outputs, 'train')
        self.log_dict(metrics)

    def validation_epoch_end(self, outputs):
        metrics = collect_binary_classification_metrics(outputs, 'dev')
        self.log_dict(metrics)

    def test_epoch_end(self, outputs):
        metrics = collect_binary_classification_metrics(outputs, 'test')
        self.log_dict(metrics)
