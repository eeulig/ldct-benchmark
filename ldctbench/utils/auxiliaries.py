import json
import os
import pickle
from argparse import Namespace

import yaml


def load_yaml(path: str):
    """Load python object from yaml file"""
    with open(path) as file:
        content = yaml.load(file, Loader=yaml.FullLoader)
    return content


def save_yaml(content, path: str):
    """Save python object to yaml file"""
    with open(path, "w") as file:
        yaml.dump(content, file)


def load_json(path: str):
    """Load python object from json file"""
    with open(path) as f:
        d = json.load(f)
        return d


def dump_config(args: Namespace, path: str):
    """Save argparse.Namespace to yaml file"""
    with open(os.path.join(path, "args.yaml"), "w") as outfile:
        yaml.dump(vars(args), outfile, default_flow_style=False)


def save_obj(obj, path: str):
    """Save python object to pickle file"""
    with open(path, "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(path: str):
    """Load python object from pickle file"""
    with open(path, "rb") as f:
        return pickle.load(f)
