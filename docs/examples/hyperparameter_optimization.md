!!! info "Prerequisite"
    This example assumes you have:

    1. The package `ldct-benchmark` installed
    2. The LDCT dataset downloaded to a folder `path/to/ldct-data`
    3. The environment variable `LDCTBENCH_DATAFOLDER` set to that folder
    
    Please refer to [Getting Started](../getting_started.md) for instructions on how to do these steps.

## What is hyperparameter optimization?
Hyperparameter optimization is the process of finding the optimal configuration of hyperparameters for a machine learning model. In deep neural networks, hyperparameters such as learning rate, batch size, or weights of different loss terms can greatly influence model performance. Therefore, proper tuning of these parameters is crucial. The four most common methods for hyperparameter optimization are human optimization, grid search, randomized search, and Bayesian optimization.

1. **Human optimization:** Manual fine-tuning involves iteratively adjusting hyperparameters based on intuition, experience, and observation of model performance. While this approach leverages domain expertise, it is time-consuming, subjective, and often fails to explore the hyperparameter space thoroughly.

2. **Grid Search:** Grid search systematically evaluates every combination of predefined hyperparameter values. It's comprehensive but computationally expensive, especially with many hyperparameters, as the number of combinations grows exponentially.

3. **Randomized Search:** Randomized search samples hyperparameter values from specified distributions rather than exhaustively trying all combinations. It is generally preferred over grid search[^1].

4. **Bayesian Optimization:** Bayesian optimization builds a surrogate model of the objective function and uses an acquisition function to determine which hyperparameter combinations to try next. Bayesian optimization has been shown to outperform other optimization methods on a variety of DNN and dataset configurations[^2][^3].

## Define hyperparameters to optimize
As for logging, we use [Weights & Biases](https://docs.wandb.ai/tutorials/sweeps/) to tune hyperparameters. To configure a hyperparameter optimization run (sweep) you need to define a configuration file in YAML format. The configuration file specifies the optimization method, hyperparameters to tune, and their respective distributions. To optmize the CNN-10 model, we could use the following config file:


=== "configs/cnn10-hpopt.yaml"
    
    ```yaml
    project: ldct-benchmark
    program: ldctbench-train
    command:
    - ${program}
    - ${args_no_boolean_flags}
    description: Hyperparameter optimization for CNN-10
    method: bayes
    metric:
        goal: maximize
        name: SSIM
    name: cnn10-hpopt
    parameters:
    adam_b1:
        value: 0.9
    adam_b2:
        value: 0.999
    data_norm:
        value: meanstd
    data_subset:
        value: 1
    devices:
        value: 0
    dryrun:
        value: false
    iterations_before_val:
        value: 1000
    lr:
        distribution: log_uniform_values
        max: 0.01
        min: 1e-05
    max_iterations:
        distribution: int_uniform
        max: 100000
        min: 1000
    mbs:
        distribution: int_uniform
        max: 128
        min: 2
    num_workers:
        value: 4
    optimizer:
        value: adam
    patchsize:
        distribution: int_uniform
        max: 128
        min: 32
    seed:
        value: 1332
    trainer:
        value: cnn10
    valsamples:
        value: 8
    ```

For your model, you may need to adjust the hyperparameters and their distributions. All arguments defined in the `argparser.py` of your method should be set in this configuration file either to a fixed value (not optimized) or a distribution of values (optimized). A full list of hyperparameter configuration options can be found in the [W&B documentation](https://docs.wandb.ai/guides/sweeps/sweep-config-keys/).

## Run hyperparameter optimization
Once you defined the configuration file, you can run hyperparameter optimization using the `ldctbench-hpopt` command. The command takes two arguments: the path to the configuration file and the number of runs to execute. The number of runs determines how many times the optimization algorithm will sample hyperparameters and train the model. The more runs you specify, the more likely you are to find the optimal hyperparameters.

To run ten steps of hyperparameter optimization using the configuration file described above, execute the following command:

```bash
ldctbench-hpopt --config configs/cnn10-hpopt.yaml --n_runs 10
```

After all runs are completed, this script reports the best run(1). This run can be used as `--methods` argument to the [test script][evaluate-custom-trained-models]. You can also visualize the results in the W&B dashboard by logging in to your W&B account and navigating to the respective project/sweep.
{ .annotate }

1.  What is *best* is determined from the `metric` parameter in the config file. For the example above it would be the one that maximizes the SSIM on the validation data.


[^1]: Bergstra, James, and Yoshua Bengio. 2012. “Random Search for Hyper-Parameter Optimization.” Journal of Machine Learning Research 13 (10): 281–305.
[^2]: Bergstra, James, Rémi Bardenet, Yoshua Bengio, and Balázs Kégl. 2011. “Algorithms for Hyper-Parameter Optimization.” In . Vol. 24.
[^3]: Snoek, Jasper, Hugo Larochelle, and Ryan P Adams. 2012. “Practical Bayesian Optimization of Machine Learning Algorithms.” In Advances in Neural Information Processing Systems. Vol. 25.

