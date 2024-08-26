!!! info "Prerequisite"
    This example assumes you have:

    1. the package `ldct-benchmark` installed
    2. the LDCT dataset downloaded to a folder `path/to/ldct-data`
    3. The environment variable `LDCTBENCH_DATAFOLDER` set to that folder
    
    Please refer to [Getting Started](../getting_started.md) for instructions on how to do these steps.

## Test script 
Evaluating some model on the test data split of the LDCT data is as simple as running `ldctbench-test` with the following arguments:
```
usage: ldctbench-test [-h] [--methods METHODS [METHODS ...]] [--metrics METRICS [METRICS ...]] [--datafolder DATAFOLDER] [--data_norm DATA_NORM] [--results_dir RESULTS_DIR] [--device DEVICE] [--print_table]

optional arguments:
  -h, --help            show this help message and exit
  --methods METHODS [METHODS ...]
                        Methods to evaluate. Can be either method name of pretrained networks (cnn10, redcnn, qae, ...), or run name of network trained by the user.
  --metrics METRICS [METRICS ...]
                        Metrics to compute. Must be either SSIM, PSNR, RMSE, or VIF.
  --datafolder DATAFOLDER
                        Path to datafolder. If not set, an environemnt variable LDCTBENCH_DATAFOLDER must be set.
  --data_norm DATA_NORM
                        Input normalization: Must be either minmax or meanstd.
  --results_dir RESULTS_DIR
                        Folder where to store results.
  --device DEVICE       CUDA device id.
  --print_table         Print results in table.
```
The argument `--datafolder` can be omitted if the environment variable `LDCTBENCH_DATAFOLDER` is set accordingly (see [here][3-set-environment-variable-to-the-data-folder]).


## Evaluate pretrained models
To test the pretrained CNN-10 and measure SSIM and PSNR, we would run:

```shell
ldctbench-test --methods cnn10 --metrics SSIM PSNR --datafolder path/to/ldct-data
```
or 
```shell
ldctbench-test --methods cnn10 --metrics SSIM PSNR
```
if the environment variable `LDCTBENCH_DATAFOLDER` is set.

By default, results are stored in `./results/test_metrics.yaml`. To print them in a markdown table, we can add `--print_table`. To reproduce [this table][test-set-performance], we can run:
```shell
ldctbench-test --datafolder path/to/ldct-data --print_table
```

## Evaluate custom trained models
To test a custom trained model we can provide the wandb run name as `--method` instead if the `wandb` folder with training logs is in the same directory from which `ldctbench-test` is called.

Evaluating the model trained in the [previous example](train_custom_model.md) with:
```shell
ldctbench-test --methods offline-run-<timestamp> --metrics PSNR SSIM --print_table
```
will print the following table

| Method                   |   PSNR (Chest) |   PSNR (Abdomen) |   PSNR (Neuro) |   SSIM (Chest) |   SSIM (Abdomen) |   SSIM (Neuro) |
|--------------------------|----------------|------------------|----------------|----------------|------------------|----------------|
| LD                       |         18.066 |           29.117 |         30.923 |          0.312 |            0.856 |          0.914 |
| offline-run-<timestamp\> |         27.036 |           32.552 |         32.15  |          0.552 |            0.904 |          0.925 |

As we can see the simple network improved over the LD baseline on all anatomies and metrics.

