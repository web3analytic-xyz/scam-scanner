from os.path import join
from ..src.paths import HUB_DIR

MODELS = {
    'pretrained-12-22': 'pretrained-12-22.ckpt',
}

def get_model(name):
    return join(HUB_DIR, MODELS[name])
