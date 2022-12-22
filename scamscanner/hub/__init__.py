from os.path import join
from ..src.paths import HUB_DIR

MODELS = {
    'pretrained-12-22': 'pretrained-12-22.ckpt',
    'featurizer-12-22': 'featurizer-12-22.joblib',
}

def get_model(name):
    return join(HUB_DIR, MODELS[name])
