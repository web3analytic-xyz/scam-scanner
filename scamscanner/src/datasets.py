import torch
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from os.path import join
from torch.utils.data import Dataset, DataLoader

from .paths import DATA_DIR


def build_loaders(batch_size, num_workers=0, rs=None):
    r"""Create data loaders on Bag of Words datasets."""

    train_dset = BagOfWordsDataset(split='train', rs=rs)
    test_dset = BagOfWordsDataset(split='test', featurizer=train_dset.featurizer, rs=rs)

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


class BagOfWordsDataset(Dataset):
    r"""Bag-of-words dataset over op-codes.
    Notes:
    --
    Actually, we use TF-IDF features
    """
    def __init__(self, split='train', featurizer=None, rs=None):
        super().__init__()

        data = pd.read_csv(join(DATA_DIR, f'processed/{split}.csv'))
        data = data.reset_index(drop=True)

        if featurizer is None:
            featurizer = TfidfVectorizer()
            featurizer.fit(data['opcode'])

        feats = featurizer.transform(data['opcode']).toarray()
        labels = data['label']
        
        if rs is None:
            rs = np.random.RandomState(42)

        feats, labels = self.balance(feats, labels, rs)

        self.feats = feats
        self.labels = labels
        self.featurizer = featurizer

    def balance(self, X, y, rs):
        pos_indices = np.where(y == 1)[0]
        neg_indices = np.where(y == 0)[0]

        pos_feats = X[pos_indices]
        neg_indices = rs.choice(neg_indices, size=len(pos_indices), replace=False)
        neg_feats = X[neg_indices]

        X = np.concatenate([pos_feats, neg_feats])
        y = np.concatenate([np.ones(len(pos_feats)), np.zeros(len(pos_feats))])

        return X, y

    def __getitem__(self, index):
        result = {
            'feat': torch.from_numpy(self.feats[index]).float(),
            'label': int(self.labels[index])
        }
        return result

    def __len__(self):
        return len(self.feats)
