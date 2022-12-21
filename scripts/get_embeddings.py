import torch

import pandas as pd
from tqdm import tqdm
from os.path import join
from scamscanner.src.paths import DATA_DIR

from transformers import LongformerModel, LongformerTokenizer


def embed_dataset(dataset, model, tokenizer, batch_size=8):
    r"""Embed a dataset by passing it through a tokenizer and a pretrained 
    transformer model.
    """
    batch_input_ids = []
    batch_attention_masks = []
    all_outputs = []

    for i in tqdm(range(len(dataset)), desc='embedding...'):
        row = dataset.iloc[i]
        opcode = row['opcode']
        encoded = tokenizer(opcode)

        batch_input_ids.append(encoded['input_ids'])
        batch_attention_masks.append(encoded['attention_mask'])

        if len(batch_input_ids) == batch_size:
            batch_input_ids = torch.stack(batch_input_ids)
            batch_attention_masks = torch.stack(batch_attention_masks)

            outputs = model(batch_input_ids, attention_mask=batch_attention_masks)
            sequence_output = outputs.last_hidden_state
            all_outputs.append(sequence_output)

    if len(batch_input_ids) > 0:
        batch_input_ids = torch.stack(batch_input_ids)
        batch_attention_masks = torch.stack(batch_attention_masks)

        outputs = model(batch_input_ids, attention_mask=batch_attention_masks)
        sequence_output = outputs.last_hidden_state
        all_outputs.append(sequence_output)

    all_outputs = torch.stack(all_outputs)
    return all_outputs


@torch.no_grad()
def main():
    r"""Tokenize the OPCODES and embed them using a Pretrained Longformer."""

    train_data = pd.read_csv(join(DATA_DIR, 'processed/train.csv'))
    test_data = pd.read_csv(join(DATA_DIR, 'processed/test.csv'))

    # Load a tokenizer
    model = LongformerModel.from_pretrained("allenai/longformer-base-4096")
    tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')

    train_emb = embed_dataset(train_data, model, tokenizer)
    torch.save(train_emb, join(DATA_DIR, 'processed/train.pt'))

    # Free some memory
    del train_emb
    del train_data

    test_emb = embed_dataset(test_data, model, tokenizer)
    torch.save(test_emb, join(DATA_DIR, 'processed/test.pt'))


if __name__ == "__main__":
    main()
