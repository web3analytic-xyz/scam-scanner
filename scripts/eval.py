import pytorch_lightning as pl

from scamscanner.src.utils import seed_everything
from scamscanner.src.datasets import build_loaders
from scamscanner.src.models import ScamScanner


def main(args):
    devices = [int(x) for x in args.devices.split(',')]

    # Load the module from a checkpoint
    module = ScamScanner.load_from_checkpoint(args.checkpoint_path)
    module.eval()

    # Fetch the config from the module
    config = module.config

    # Fix the random seeds for reproducibility
    rs = seed_everything(config.machine.seed, use_cuda=config.machine.use_cuda)

    # Build the data loader
    _, test_loader = build_loaders(config.optimizer.batch_size, config.machine.num_workers, rs=rs)

    # Run through the test set and get the loss
    trainer = pl.Trainer(default_root_dir=config.experiment.exp_dir,
                         accelerator='gpu',
                         devices=devices,
                         )
    trainer.test(model=module, dataloaders=test_loader)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint_path', type=str, help='Path to trained checkpoint file')
    parser.add_argument(
        '--devices',
        type=str,
        default='0',
        help='GPU device (default: 0). If specifying multiple, use commas',
    )
    args = parser.parse_args()

    main(args)
