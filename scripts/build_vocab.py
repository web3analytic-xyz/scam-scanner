import json
import pandas as pd
from os.path import join
from tqdm import tqdm
from collections import defaultdict

from scamscanner.src.paths import DATA_DIR

def main():
    data = pd.read_csv(join(DATA_DIR, 'train.csv'))
    vocab = defaultdict(lambda: 0)
    for i in tqdm(range(len(data)), desc='building vocab'):
        opcode = data.iloc[i]['opcode']
        opcode = opcode.split()
        for op in opcode:
            vocab[op] += 1

    vocab = dict(vocab)
    out_file = join(DATA_DIR, 'train-vocab.json')
    with open(out_file, 'w') as fp:
        json.dump(vocab, fp)


if __name__ == "__main__":
    main()
