import h5py
import torch

import pandas as pd
from tqdm import tqdm
from os.path import join
from scamscanner.src.paths import DATA_DIR

from transformers import LongformerModel, LongformerTokenizer


def embed_dataset(dataset, model, tokenizer, device, out_file, batch_size=8):
    r"""Embed a dataset by passing it through a tokenizer and a pretrained 
    transformer model.
    """
    with h5py.File(out_file, 'w') as hf:
        batch_names = []
        batch_input_ids = []
        batch_attention_masks = []

        for i in tqdm(range(len(dataset)), desc='embedding...'):
            row = dataset.iloc[i]
            opcode = row['opcode']
            encoded = tokenizer(opcode)

            batch_names.append(row['address'])
            batch_input_ids.append(encoded['input_ids'])
            batch_attention_masks.append(encoded['attention_mask'])

            if len(batch_input_ids) == batch_size:
                batch_input_ids = torch.stack(batch_input_ids).to(device)
                batch_attention_masks = torch.stack(batch_attention_masks).to(device)

                outputs = model(batch_input_ids, attention_mask=batch_attention_masks)
                sequence_output = outputs.last_hidden_state

                for j in range(batch_size):
                    hf.create_dataset(batch_names[j], data=sequence_output[j].cpu().numpy())

        if len(batch_names) > 0:
            batch_input_ids = torch.stack(batch_input_ids)
            batch_attention_masks = torch.stack(batch_attention_masks)

            outputs = model(batch_input_ids, attention_mask=batch_attention_masks)
            sequence_output = outputs.last_hidden_state

            for j in range(len(batch_names)):
                hf.create_dataset(batch_names[j], data=sequence_output[j].cpu().numpy())


@torch.no_grad()
def main(args):
    r"""Tokenize the OPCODES and embed them using a Pretrained Longformer."""

    train_data = pd.read_csv(join(DATA_DIR, 'processed/train.csv'))
    test_data = pd.read_csv(join(DATA_DIR, 'processed/test.csv'))

    # Reset the index
    train_data = train_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)

    device = torch.device(f'cuda:{args.device}')

    # Load a tokenizer & model
    model = LongformerModel.from_pretrained("allenai/longformer-base-4096")
    tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')

    # Cast to device
    model = model.to(device)
    tokenizer = tokenizer.to(device)

    train_file = join(DATA_DIR, 'processed/train.h5')
    embed_dataset(train_data, model, tokenizer, device, train_file)

    # Free some memory
    del train_data

    test_file = join(DATA_DIR, 'processed/test.h5')
    embed_dataset(test_data, model, tokenizer, device, test_file)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    main(args)
