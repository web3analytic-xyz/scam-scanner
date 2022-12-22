import pytorch_lightning as pl

from scamscanner.src.utils import seed_everything, process_config
from scamscanner.src.datasets import build_loaders_bow
from scamscanner.src.models import ScamScanner_BoW


def main(args):
    config = process_config(args.config_path)
    devices = sorted([int(x) for x in args.devices.split(',')])

    # Fix the random seeds for reproducibility
    seed_everything(config.machine.seed, use_cuda=config.machine.use_cuda)

    # Build the data loaders
    train_loader, dev_loader = build_loaders_bow(config.optimizer.batch_size, config.machine.num_workers)

    # Load the module we wish to use
    module = ScamScanner_BoW(config)

    # Save the checkpoint weights by minimum dev loss
    checkpoint_callback = pl.callbacks.ModelCheckpoint(config.checkpoint_dir, every_n_epochs=5)

    # Create a trainer instance
    trainer = pl.Trainer(
        default_root_dir=config.experiment.exp_dir,
        precision=32,
        max_epochs=config.optimizer.max_epochs,
        callbacks=[checkpoint_callback],
        accelerator='gpu',
        devices=devices,
        gradient_clip_val=config.optimizer.clip_grad_norm,
    )

    # Call fit to train the module
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=dev_loader)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str, help='Path to YAML config file')
    parser.add_argument(
        '--devices', 
        type=str,
        default='0', 
        help='GPU device (default: 0). If specifying multiple, use commas',
    )
    args = parser.parse_args()

    main(args)
