import torch
import pandas as pd
import numpy as np

from os.path import join
from torch.utils.data import Dataset, DataLoader
from transformers import LongformerTokenizer

from .paths import DATA_DIR


def build_loaders():
    pass


class ContractDataset(Dataset):
    r"""Dataset of contracts with labels.
    Notes:
    --
    Tokenizes OPCODES with Longformer.
    """
    def __init__(self, split='train'):
        super().__init__()

        data = pd.read_csv(join(DATA_DIR, f'{split}.csv'))
        data = data.reset_index(drop=True)
        data = self.balance(data)

        self.data = data
        self.tokenizer = tokenizer

    def balance(self, data):
        data['label']

    def __getitem__(self, index):

        opcode = row['opcode']
        tokens = self.tokenizer(opcode)
        label = int(row['label'])

        return tokens, label

    def __len__(self):
        return min(len(self.pos_indices), len(self.neg_indices)) * 2
