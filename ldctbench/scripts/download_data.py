import argparse
import io
import os
import warnings
import zipfile

import pandas as pd
import requests
from tqdm import tqdm

BASE_URL = "https://services.cancerimagingarchive.net/nbia-api/services/v2/"


def get_series(manifest_file):
    # Similar to tcia_utils.nbia.manifestToList()
    data = []
    with open(manifest_file) as f:
        for line in f:
            data.append(line.rstrip())
    del data[:6]  # remove header
    return data


def get_token(user, pwd):
    # Similar to tcia_utils.nbia.getToken()
    global api_call_headers
    try:
        params = {
            "client_id": "nbia",
            "scope": "openid",
            "grant_type": "password",
            "username": user,
            "password": pwd,
        }

        data = requests.post(
            "https://keycloak-stg.dbmi.cloud/auth/realms/TCIA/protocol/openid-connect/token",
            data=params,
        )
        data.raise_for_status()
        access_token = data.json()["access_token"]
        api_call_headers = {"Authorization": "Bearer " + access_token}

    # handle errors
    except requests.exceptions.HTTPError as errh:
        raise ValueError(
            f"HTTP Error: {data.status_code} -- Double check your user name and password."
        )
    except requests.exceptions.ConnectionError as errc:
        raise ValueError(f"Connection Error: {data.status_code}")
    except requests.exceptions.Timeout as errt:
        raise ValueError(f"Timeout Error: {data.status_code}")
    except requests.exceptions.RequestException as err:
        raise ValueError(f"Request Error: {data.status_code}")


def download_series(series: str, savedir: str):
    # Similar to tcia_utils.nbia.downloadSeries()
    global metadata_df

    data_url = f"{BASE_URL}getImage?NewFileNames=Yes&SeriesInstanceUID={series}"
    metadata_url = f"{BASE_URL}getSeriesMetaData?SeriesInstanceUID={series}"

    # Get metadata
    metadata = requests.get(metadata_url, headers=api_call_headers).json()
    if (
        "Series UID" in metadata_df.columns
        and series in metadata_df["Series UID"].to_list()
    ):
        warnings.warn(f"Skip {series} as it was already downloaded")
        return

    # Construct folder path
    m = metadata[0]
    series_savedir = os.path.join(
        savedir,
        "LDCT-and-Projection-data",
        m["Subject ID"],
        f"{m['Study Date']}-NA-NA-{m['Study UID'][-5:]}",
        f"{m['Series Number']}-{m['Series Description']}-{m['Series UID'][-5:]}",
    )

    if not os.path.exists(series_savedir):
        os.makedirs(series_savedir)
    else:
        # Delete anything in the folder in case of a previous partial download
        for file in os.listdir(series_savedir):
            os.remove(os.path.join(series_savedir, file))

    # Download data
    data = requests.get(data_url, headers=api_call_headers)
    file = zipfile.ZipFile(io.BytesIO(data.content))
    file.extractall(path=series_savedir)

    # Update metadata
    metadata_df = pd.concat([metadata_df, pd.DataFrame(metadata)], ignore_index=True)


def main():
    # This script can be used to download the LDCT and Projection data using tcia_utils.
    # Some of the code herein is heavily inspired by the tcia-utils python package
    # (https://github.com/kirbyju/tcia_utils). We do this to reduce package dependencies
    # (tcia_utils wants a lot of packages we don't need here), show progress bars, and
    # store data in same folder structure as the nbia-data-retriever would do.
    global metadata_df
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="", help="nbia manifest file")
    parser.add_argument(
        "--savedir", default="ldct-data", help="Folder to which data is downloaded"
    )
    parser.add_argument("--username", default="", help="TCIA username")
    parser.add_argument("--password", default="", help="TCIA password")
    opt = parser.parse_args()

    if not opt.manifest:
        opt.manifest = os.path.join("assets", "manifest.tcia")

    assert opt.username, "Must provide username since LDCT data has restricted access!"
    assert opt.password, "Must provide password since LDCT data has restricted access!"

    # Get list of series we want to download
    series_to_download = get_series(opt.manifest)

    # Get token since data access is restricted
    get_token(user=opt.username, pwd=opt.password)

    # Create metadata csv
    metadata_path = os.path.join(opt.savedir, "metadata.csv")
    if os.path.isfile(metadata_path):
        print("Load existing metadata.csv")
        metadata_df = pd.read_csv(metadata_path)
    else:
        metadata_df = pd.DataFrame()

    for series in tqdm(series_to_download, desc="Download LDCT data"):
        # Download data
        download_series(series=series, savedir=opt.savedir)
        # Update manifest file
        metadata_df.to_csv(metadata_path)


if __name__ == "__main__":
    main()
