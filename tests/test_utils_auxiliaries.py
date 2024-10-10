import os
import tempfile
from argparse import Namespace

import pytest

from ldctbench.utils.auxiliaries import (
    dump_config,
    load_json,
    load_obj,
    load_yaml,
    save_json,
    save_obj,
    save_yaml,
)


@pytest.fixture(scope="session")
def tempdir():
    tempdir = tempfile.TemporaryDirectory()
    yield tempdir.name
    tempdir.cleanup()


def test_save_load_yaml_returns_correct_content(tempdir):
    content = {"a": 1, "b": [1, 2, 3], "c": "abc"}
    save_yaml(content, os.path.join(tempdir, "test.yaml"))
    loaded_content = load_yaml(os.path.join(tempdir, "test.yaml"))
    assert content == loaded_content


def test_save_load_json_returns_correct_content(tempdir):
    content = {"a": 1, "b": [1, 2, 3], "c": "abc"}
    save_json(content, os.path.join(tempdir, "test.json"))
    loaded_content = load_json(os.path.join(tempdir, "test.json"))
    assert content == loaded_content


def test_dump_config(tempdir):
    args = Namespace(a=1, b=[1, 2, 3], c="abc")
    dump_config(args, tempdir)
    loaded_args = load_yaml(os.path.join(tempdir, "args.yaml"))
    assert args == Namespace(**loaded_args)


def test_save_load_object_returns_correct_content(tempdir):
    content = {"a": 1, "b": [1, 2, 3], "c": "abc"}
    save_obj(content, os.path.join(tempdir, "test.pkl"))
    loaded_content = load_obj(os.path.join(tempdir, "test.pkl"))
    assert content == loaded_content
