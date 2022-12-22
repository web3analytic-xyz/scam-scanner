from os.path import join, realpath


SRC_DIR = realpath(join(__file__, '..'))
ROOT_DIR = realpath(join(SRC_DIR, '..'))
DATA_DIR = join(ROOT_DIR, 'data')
CONFIG_DIR = join(ROOT_DIR, 'configs')
HUB_DIR = join(ROOT_DIR, 'hub')
