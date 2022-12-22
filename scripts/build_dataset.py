from os.path import join
from os import makedirs, environ

import time
import pickle
import numpy as np
from tqdm import tqdm

from scamscanner.src.utils import (
    load_data,
    get_contract_code,
    split_data,
    get_w3,
)
from scamscanner.src.paths import DATA_DIR


def main():
    r"""Build a dataset of scam and non-scam contracts. For each, we need to pull
    the raw OPCODES, and a label. This is saved as a dataframe.
    """
    data = load_data()
    w3 = get_w3()

    # Fix the random seed
    rs = np.random.RandomState(42)

    out_dir = join(DATA_DIR, 'processed')
    makedirs(out_dir, exist_ok=True)

    # We expect this to exist!
    etherscan_api_key = environ['ETHERSCAN_API_KEY']

    abi, bytecode, opcode = [], [], []

    for i in tqdm(range(len(data)), desc='Getting contract data'):
        row = data.iloc[i]
        output = get_contract_code(row['contract_address'], etherscan_api_key, w3)

        if output is None:
            abi.append(np.nan)
            opcode.append(np.nan)
            bytecode.append(np.nan)
        else:
            abi.append(output['abi'])
            opcode.append(output['opcode'])
            bytecode.append(output['bytecode'])

        time.sleep(0.25)

        if (i+1) % 500 == 0:
            cache = {'abi': abi, 'bytecode': bytecode, 'opcode': opcode}
            with open(join(out_dir, 'cache.pkl'), 'wb') as fp:
                pickle.dump(cache, fp)
            print(f'Cached {i+1} entries.')

    data['abi'] = abi
    data['opcode'] = opcode
    data['bytecode'] = bytecode

    breakpoint()

    # Drop all the data that has missing entries
    data = data.dropna()

    train_data, test_data = split_data(data, rs=rs)

    train_data.to_csv(join(out_dir, 'train.csv'), index=False)
    test_data.to_csv(join(out_dir, 'test.csv'), index=False)


if __name__ == "__main__":
    main()
