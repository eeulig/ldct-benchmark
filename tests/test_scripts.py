import os


def test_entrypoint_download_data_is_found():
    exit_status = os.system("ldctbench-download-data --help")
    assert exit_status == 0


def test_entrypoint_train_is_found():
    exit_status = os.system("ldctbench-train --help")
    assert exit_status == 0


def test_entrypoint_test_is_found():
    exit_status = os.system("ldctbench-test --help")
    assert exit_status == 0
