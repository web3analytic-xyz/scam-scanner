import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PreprocessNet(nn.Module):
    r"""Network for preprocessing a feature vector into a shared shape.
    Arguments:
    --
    input_dim: integer
        Number of input dimensions
    output_dim: integer
        Number of output dimensions
    dropout_prob: float (default: 0.5)
        Probability of dropout
    """

    def __init__(self, input_dim, output_dim, dropout_prob=0.5):
        super().__init__()
        self.layernorm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.linear = nn.Linear(input_dim, output_dim * 2)  # x 2 for GLU activation

    def forward(self, x):
        x = self.layernorm(x)
        x = self.dropout(x)
        x = self.linear(x)
        x = F.glu(x)
        return x


class Conv1d_Norm(nn.Module):
    r"""Combines convolution and layer normalization.
    Arguments:
    --
    input_dim: integer
        Number of input dimensions
    output_dim: integer
        Number of output dimensions
    kernel_size: integer
        Kernel size for convolution layer
    dilation: integer (default: 1)
        Dilation factor for convolution layer
    activation: string (default: None)
        Nonlinearity to apply at the end
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        kernel_size,
        dilation=1,
        activation=None,
    ):
        super().__init__()
        self.layernorm = nn.LayerNorm(input_dim)
        self.conv = nn.Conv1d(
            in_channels=input_dim,
            out_channels=output_dim,
            kernel_size=kernel_size,
            padding=((kernel_size - 1) * dilation) // 2,
            dilation=dilation,
        )
        self._activation = activation

    def forward(self, x):
        # `x` shape = (batch_size, num_frames, num_channels)
        x = self.layernorm(x)
        # `conv` input shape = (batch_size, num_channels, num_frames)
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        x = apply_activation(x, self._activation)
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


class GRU_Norm(nn.Module):
    r"""Combines GRU and layer normalization.
    Arguments:
    --
    input_dim: integer
        Number of input dimensions
    output_dim: integer
        Number of output dimensions
    bidirectional: boolean (default: False)
        Is the GRU bi-directional? 
    """
    def __init__(self, input_dim, output_dim, bidirectional=False):
        super().__init__()
        self.layernorm = nn.LayerNorm(input_dim)
        self.gru = nn.GRU(input_dim, output_dim, batch_first=True, 
                           bidirectional=bidirectional)
    def forward(self, x, h=None):
        x = self.layernorm(x)
        x, h = self.gru(x, h)
        return x, h


class ConvTranspose1d_Norm(nn.Module):
    r"""Combines a transposed convolution layer and layer normalization."""

    def __init__(self,
                 input_dim,
                 output_dim,
                 kernel_size=3,
                 stride=2,
                 padding=1,
                 ):
        super().__init__()

        self.layernorm = nn.LayerNorm(input_dim)
        self.convtranspose = nn.ConvTranspose1d(in_channels=input_dim,
                                                out_channels=output_dim,
                                                kernel_size=kernel_size, 
                                                stride=stride,
                                                padding=padding,
                                                )

    def forward(self, x):
        x = self.layernorm(x)
        # conv input shape = (batch, channels, frames)
        x = self.convtranspose(x.transpose(1,2)).transpose(1,2)
        
        return x


class PaddedAveragePooling(nn.Module):
    r"""Average pooling with contextual padding.
    Arguments:
    --
    lookback: integer
        Amount of left padding 
    lookahead: integer
        Amount of right padding 
    activation: string (default: None)
        Nonlinearity to apply at the end
    """
    def __init__(self, lookback, lookahead, activation=None):
        super().__init__()
        self.pool = nn.AvgPool1d(kernel_size=lookback+lookahead, stride=1, padding=0)
        self._activation = activation
        self._lookback = lookback
        self._lookahead = lookahead

    def forward(self, x):
        # `x` shape = (batch_size, num_frames, num_channels)
        pad = (0, 0, self._lookback - 1, self._lookahead)
        x = F.pad(x, pad, 'constant', 0)
        # `x` shape = (batch_size, num_frames * 2, num_channels)
        x = self.pool(x.transpose(1, 2)).transpose(1, 2)
        x = apply_activation(x, self._activation)
        return x


class PositionalEncoding(nn.Module):
    r"""Positional encoding for transformer networks.
    Arguments:
    --
    d_model: integer
        Size of the input vector.
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, shift=0):
        x = x + self.pe[:, shift:shift+x.size(1), :]
        return self.dropout(x)

    def inference(self, x, ts):
        x = x + self.pe[:, ts, :]
        return self.dropout(x)


class PostprocessNet(nn.Module):
    r"""PostprocessNet to take in all the predicted Mel spectrograms and learn a 
    residual factor. 
    Notes:
    --
    Five 1-d convolution with 512 channels and kernel size 5.
    There is some work to be done to make this usable live.
    """
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 kernel_size,
                 dropout_prob=0.1,
                 num_conv=5,
                 ):
        super(PostprocessNet, self).__init__()
        net = [
            Conv1d_Norm(input_dim,
                        hidden_dim,
                        kernel_size=kernel_size,
                        dilation=1,
                        activation='tanh',
                        ),
            nn.Dropout(p=dropout_prob),
        ]
        for _ in range(1, num_conv - 1):
            net.extend([
                Conv1d_Norm(hidden_dim,
                            hidden_dim,
                            kernel_size=kernel_size,
                            dilation=1,
                            activation='tanh',
                            ),
                nn.Dropout(p=dropout_prob),
            ])
        net.append(
            Conv1d_Norm(hidden_dim,
                        input_dim,
                        kernel_size=kernel_size,
                        dilation=1,
                        ),
        )
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


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
        return F.ReLU(x)
    elif activation == 'softplus':
        return F.softplus(x)
    elif activation == 'softmax':
        return F.softmax(x, dim=-1)
    elif activation == 'norm':
        return F.normalize(x, dim=-1)
    else:
        return x
