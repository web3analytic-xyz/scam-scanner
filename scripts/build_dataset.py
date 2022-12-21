from os.path import join
from os import makedirs

from scamscanner.src.utils import (
    load_data,
    get_contract_code,
    bytecode_to_opcode,
    split_data,
)
from scamscanner.src.paths import DATA_DIR


def main():
    r"""Build a dataset of scam and non-scam contracts. For each, we need to pull
    the raw OPCODES, and a label. This is saved as a dataframe.
    """
    data = load_data()
    data = get_contract_code(data)
    data = bytecode_to_opcode(data)
    train_data, test_data = split_data(data)

    out_dir = join(DATA_DIR, 'processed')
    makedirs(out_dir)

    train_data.to_csv(join(out_dir, 'train.csv'), index=False)
    test_data.to_csv(join(out_dir, 'test.csv'), index=False)


if __name__ == "__main__":
    main()
