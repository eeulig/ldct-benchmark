# Getting started

## Installation
Set up a new environment with Python 3.10 or higher. We recommend using a virtual environment to avoid conflicts with other packages.

### From PyPI (coming soon)
```
pip install ldct-benchmark
```

### From GitHub

1. Clone this repository: `git clone https://github.com/eeulig/ldct-benchmark`
2. Install the package with `pip install .`

#### In editable mode

If you want to implement your own methods or contribute to the project, you should install the package in editable mode using the `-e`flag:

1. Clone this repository: `git clone https://github.com/eeulig/ldct-benchmark`
2. Install the package with `pip install -e .`

### Dependencies

- We recommend to install the correct PyTorch version for your operating system and CUDA version from [PyTorch](https://pytorch.org/get-started/locally/){:target="_blank"} directly.
- [Only if you want to use `bilateral`] Install the trainable bilateral filter by running `pip install bilateralfilter_torch`. For details on the bilateral filter see the [official repository](https://github.com/faebstn96/trainable-bilateral-filter-source){:target="_blank"}.


## Download the LDCT data
!!! info
    This is only necessary if you want to train or test the models on the LDCT data. If you only want to apply the models to your own data, you can skip this step.
    
Our benchmark uses the Low Dose CT and Projection data[^1] which is available from the [Cancer Imaging Archive (TCIA)](https://www.cancerimagingarchive.net/collection/ldct-and-projection-data/){:target="_blank"}. For downloading the data, please follow the instructions below.


[^1]: C. McCollough, B. Chen, D. R. Holmes III, X. Duan, Z. Yu, L. Yu, S. Leng, and J. Fletcher, “Low dose CT image and projection data”, 2020.

### 1. Sign a TCIA license agreement
You must sign and submit a TCIA Restricted License Agreement to download the data. Information on how to do this is provided under "Data Access" [here](https://www.cancerimagingarchive.net/collection/ldct-and-projection-data/){:target="_blank"}.

### 2. Download the LDCT data
Download **Version 3** of the LDCT and Projection Data. We provide the `.tcia` object containing only the Siemens image-domain data (~27 GB) in `assets/manifest.tcia`.

#### Using a script (recommended)
We provide a script to download the data in `download_data.py`. Run the following command to download the data to `/path/to/datafolder`. You must provide your TCIA username and password to access the data:

```sh
python download_data.py --savedir /path/to/datafolder --username <username> --password <password>
```
!!! info
    If your username or password contains special characters, you may need to enclose them in single quotes:
    ```sh
    --username '#!fancy-username' --password 'p@$$w0rd'
    ```

#### Using the NBIA Data Retriever
##### Ubuntu 
After installing the nbia-data-retriever from [here](https://wiki.cancerimagingarchive.net/display/NBIA/Version+4.4){:target="_blank"}, the following command will download the data to `/path/to/datafolder`: 
```
/opt/nbia-data-retriever/nbia-data-retriever --cli path/to/repo/assets/manifest.tcia -d /path/to/datafolder -v –f -u <username> -p <password>
```
with `<username>` and `<password>` being your TCIA username/password. Alternatively, you can use the GUI to download using the provided `manifest.tcia`.

##### Windows / Mac
After installing the nbia-data-retriever from [here](https://wiki.cancerimagingarchive.net/display/NBIA/Version+4.4){:target="_blank"} use the application to download the data using the provided `manifest.tcia`. You'll need to provide your TCIA username and password.

### 3. Set environment variable to the data folder
For training and testing the models on the LDCT data you need to set the environment variable `LDCTBENCH_DATAFOLDER` to the path of the downloaded data folder. You can do this by running the following command:
```
export LDCTBENCH_DATAFOLDER=path/to/ldct-data
```
where `path/to/ldct-data` is the path to the downloaded data folder. Alternatively, you can provide the path to the data folder in the `config.yaml` file or via the argument `--datafolder` when running the scripts.
