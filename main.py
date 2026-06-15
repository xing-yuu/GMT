import argparse
from types import SimpleNamespace

import yaml

from train.sp_lightning import train


def _namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_namespace(item) for item in value]
    return value


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return _namespace(yaml.safe_load(handle))


def main():
    parser = argparse.ArgumentParser(description="GMT Jittor trainer")
    parser.add_argument("config", type=str, help="Path to the YAML config file")
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
