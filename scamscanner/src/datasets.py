import json
import h5py
import torch
import pandas as pd
from collections import defaultdict

from os.path import join
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from .paths import DATA_DIR


def build_bow_loaders(batch_size, num_workers=0):
    r"""Create data loaders on Bag of Words datasets."""

    train_dset = BagOfWordsDataset(split='train')
    test_dset = BagOfWordsDataset(split='test')

    train_loader = DataLoader(
        train_dset,
        batch_size=batch_size,
        shuffle=True, 
        pin_memory=True,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dset,
        batch_size=batch_size,
        shuffle=False, 
        pin_memory=True,
        num_workers=num_workers,
    )

    return train_loader, test_loader


def build_emb_loaders(batch_size, num_workers=0):
    r"""Create data loaders on embedded datasets."""

    train_dset = EmbeddedDataset(split='train')
    test_dset = EmbeddedDataset(split='test')

    train_loader = DataLoader(
        train_dset,
        batch_size=batch_size,
        shuffle=True, 
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=collator,
    )
    test_loader = DataLoader(
        test_dset,
        batch_size=batch_size,
        shuffle=False, 
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=collator,
    )

    return train_loader, test_loader


def collator(list_of_dicts, padding_value=0):
    r"""Given a list of elements, combine together. For sequential data, 
    we need to pad the sequence.
    Arguments:
    --
    list_of_dicts: list[dict[str, any]]
        List of batch elements
    """
    names = list_of_dicts[0].keys()
    batch_dict = {}
    for name in names:
        tmp = [var_dict[name] for var_dict in list_of_dicts]
        batch_dict[name] = pad_sequence(tmp,
                                        batch_first=True,
                                        padding_value=padding_value,
                                        )
    return batch_dict


class BagOfWordsDataset(Dataset):
    r"""Bag-of-words dataset over op-codes."""

    def __init__(self, split='train'):
        super().__init__()

        data = pd.read_csv(join(DATA_DIR, f'{split}.csv'))
        data = data.reset_index(drop=True)

        vocab_file = join(DATA_DIR, f'train-vocab.json')
        with open(vocab_file, 'r') as fp:
            vocab = json.load(fp)

        self.data = data
        self.vocab = vocab

class EmbeddedDataset(Dataset):
    r"""Dataset of contracts with labels.
    Notes:
    --
    Loads precomputed Longformer features.
    """
    def __init__(self, split='train'):
        super().__init__()

        data = pd.read_csv(join(DATA_DIR, f'{split}.csv'))
        data = data.reset_index(drop=True)

        self.data = data
        self.emb_file = join(DATA_DIR, f'{split}.h5')

    def __getitem__(self, index):
        row = self.data.iloc[index]

        name = row['address']
        y = int(row['label'])

        # Load the precomputed embedding
        with h5py.File(self.emb_file, 'r') as hf:
            x = hf[name][:]
            x = torch.from_numpy(x).float()

        return x, y

    def __len__(self):
        return len(self.data)
