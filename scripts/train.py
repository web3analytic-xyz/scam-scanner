from os.path import join
from src.utils import seed_everything, process_config
from src.modules import Gatotron
from src.datasets import build_loaders
import pytorch_lightning as pl


def main(args):
    pass


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str, help='Path to YAML config file')
    parser.add_argument('--devices', 
                        type=str,
                        default='0', 
                        help='GPU device (default: 0). If specifying multiple, use commas',
                        )
    args = parser.parse_args()

    main(args)