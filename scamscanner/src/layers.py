import torch
import torch.nn as nn
import torch.nn.functional as F


class ResPerceptron(nn.Module):
    r"""Residual Perceptron (MLP)."""

    def __init__(self, input_dim, dropout_prob=0.5):
        super().__init__()

        self.fc1 = Linear_Norm(input_dim, input_dim, activation='relu')
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc2 = Linear_Norm(input_dim, input_dim, activation='relu')
        self.fc3 = Linear_Norm(input_dim, 1)

    def forward(self, x):
        h = self.fc2(self.dropout(self.fc1(x)))
        x = self.fc3(x + h)
        return x


class Linear_Norm(nn.Module):
    r"""Combines linear and layer normalization.
    Arguments:
    --
    input_dim: integer
        Number of input dimensions
    output_dim: integer
        Number of output dimensions
    activation: string (default: None)
        Nonlinearity to apply at the end
    """
    def __init__(self, input_dim, output_dim, activation=None):
        super().__init__()

        self.layernorm = nn.LayerNorm(input_dim)
        self.linear = nn.Linear(input_dim, output_dim)
        self._activation = activation

    def forward(self, x):
        x = self.layernorm(x)
        x = self.linear(x)
        x = apply_activation(x, self._activation)
        return x


def apply_activation(x, activation):
    r"""Helper function to apply a variety of activations.
    Arguments:
    --
    x: torch.Tensor
    activation: string
        tanh | sigmoid | relu | softplus | softmax | norm | 'None'
    """
    if activation == 'tanh':
        return torch.tanh(x)
    elif activation == 'sigmoid':
        return torch.sigmoid(x)
    elif activation == 'relu':
        return F.relu(x)
    elif activation == 'softplus':
        return F.softplus(x)
    elif activation == 'softmax':
        return F.softmax(x, dim=-1)
    elif activation == 'norm':
        return F.normalize(x, dim=-1)
    else:
        return x
