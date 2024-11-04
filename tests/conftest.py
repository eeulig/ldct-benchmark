import os
import shutil
import tempfile

import pytest


class DicomData:
    def __init__(self, file_id):
        self.file_id = file_id
        self.tempdir = tempfile.TemporaryDirectory()
        self.download_dicom()
        self.folder_path = self.tempdir.name
        self.dicom_path = os.path.join(self.folder_path, f"{self.file_id}.dcm")

    def download_dicom(self):
        os.system(
            f's5cmd --no-sign-request --endpoint-url https://s3.amazonaws.com cp "s3://idc-open-data/b9cf8e7a-2505-4137-9ae3-f8d0cf756c13/{self.file_id}.dcm" {self.tempdir.name}'
        )

    def cleanup(self):
        for item in os.listdir(self.folder_path):
            if os.path.join(self.folder_path, item) == self.dicom_path:
                continue
            if os.path.isfile(os.path.join(self.folder_path, item)):
                os.remove(os.path.join(self.folder_path, item))
            else:
                shutil.rmtree(os.path.join(self.folder_path, item))

    def destroy(self):
        self.tempdir.cleanup()


@pytest.fixture(scope="session")
def sample_dicom():
    dicom_data = DicomData("496788de-f0f0-41fd-b19a-6da82268fd0a")
    yield dicom_data
    dicom_data.destroy()
