import os

import yaml

from configs import apply_config_defaults


def load_config(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        cfg = {}
    return apply_config_defaults(cfg)
