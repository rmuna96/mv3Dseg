import json
import os, errno
from munch import DefaultMunch


def sureDir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
            pass

    else:
        pass


def load_conf_file(config_file):
   if 'json' in str(config_file):
       with open(config_file, 'r') as f:
           cfg = json.load(f)
       Conf = object()
       cf = DefaultMunch.fromDict(cfg, Conf)
       return cf