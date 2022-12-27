import joblib
import numpy as np
import pandas as pd
from os.path import join

from scamscanner.src.models import ScamScanner
from scamscanner.src.layers import LogisticRegression


def main(args):
    r"""Get weights from a trained linear model to interpret which weights 
    are OPCODES are important to classificaiton.
    Notes:
    --
    Do not pass the residual perception waits to this function since we cannot
    interpret nonlinear models in the same manner.
    """
    # Load the module from a checkpoint
    module = ScamScanner.load_from_checkpoint(args.checkpoint_path)
    module.eval()

    # Fetch the config from the module
    config = module.config

    # Load pretrained featurizer
    featurizer = joblib.load(join(config.experiment.exp_dir, 'featurizer.joblib'))

    # Get the vocabulary from the featurizer
    # See https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.htmls
    vocab = featurizer.get_feature_names_out()

    assert isinstance(module.model, LogisticRegression), \
        "Only `LogisticRegression` models can be interpreted."

    model = module.model
    # shape after squeeze: `|OP_CODES|`
    params = model.fc1.linear.weight.squeeze(0).cpu().numpy()

    # Negative weights are as important as positive ones
    importance = np.abs(params)

    result = []

    order = np.argsort(importance)[::-1]
    for rank, i in enumerate(order):
        entry = {'rank': rank, 'importance': importance[i], 'opcode': vocab[i]}
        result.append(entry)

    result = pd.DataFrame.from_records(result)
    print(result)

    out_file = join(config.experiment.out_dir, 'interpret.csv')
    result.to_csv(out_file, index=False)
    print(f'Saved results to {out_file}.')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint_path', type=str, help='Path to trained checkpoint file')
    args = parser.parse_args()

    main(args)
