# Contributing 

Thank you for your interest in contributing! Here are some guidelines:

## Installation
Install the package locally using pip and in interactive mode. This way, you can immediately test your changes to the codebase.
```bash
pip install -e .[dev]
```

## Contributing denoising algorithms
1. Create a branch `git checkout -b method/fancy-method` for your new method.
2. Create a folder for the new method in `ldctbench/methods`. The folder must contain the following files
    - `__init__.py`
    - `argparser.py`: Should implement a method `add_args()` that takes as input an `argparse.ArgumentParser`, adds custom arguments and returns it. If your method has an argument `fancy_arg`, then `argparser.py` should look like this:
        ```python
        from argparse import ArgumentParser


        def add_args(parser: ArgumentParser) -> ArgumentParser:
            parser.add_argument(
                "--fancy_arg", 
                type=float, 
                help="My fancy argument."
            )
            return parser

        ```
    - `network.py`: Should implement the model as `class Model(torch.nn.Module)`.
    - `Trainer.py`: Should imeplement a `Trainer` class. This class should be initialized with `Trainer(args: argparse.Namespace, device: torch.device)` and implement a `fit()` method that trains the network. A base class is provided in `methods/base.py`.
3. Add the method to `METHODS` in `argparser.py`.
4. Add the method to `docs/denoising_algorithms.md`.
5. Add a `fancy-method.yaml` config file containing all hyperparameters to `configs/`.

## Contributing other features or bug fixes
1. Create a new branch based on your change type:
    - `fix/some-fix` for bug fixes
    - `feat/some-feature` for adding new features

    ```bash
    git checkout -b <your-branch-name>
    ```

## Code Formatting and Unit Tests
Make sure your code is:
1. Readable
2. Nicely documented with docstrings in [numpy format](https://numpydoc.readthedocs.io/en/latest/format.html)
3. Follows the style guide enforced by black, isort, and flake8

For new methods or features, **always** implement some unit tests and put them in `tests/module/test_new_feature.py`. The test function names should be descriptive. A good format would be something like `test_functionname_when_given_arguments_return_x()`. A bad format would be something like `test123()`.

#### Examples
<details>
  <summary>Don't do!</summary>
  
```python
# The new "feature" we implemented
def divide(a, b):
    # This function divides stuff
    if b == 0:
        raise ZeroDivisionError("You can't divide by zero!")
    return a / b

# Test functions
def test1():
    assert divide(1, 2) == 0.5
    assert divide(5, 5) == 1.

def test2():
    assert divide(5, -2) == -2.5
    assert divide(-2, 5) == -0.4
```
</details>

<details>
  <summary>Do!</summary>
  
```python
# The new "feature" we implemented
def divide(a: int, b: int) -> float:
    """Function to divide two numbers

    Parameters
    ----------
    a : int
        The numerator
    b : int
        The denominator

    Returns
    -------
    float
        Result of division operation

    Raises
    ------
    ZeroDivisionError
        If the denominator (b) is zero
    """
    if b == 0:
        raise ZeroDivisionError("You can't divide by zero!")
    return a / b

# Test functions
import pytest
def test_divide_when_given_positive_numbers_returns_correct_result():
    assert divide(1, 2) == 0.5
    assert divide(5, 5) == 1.

def test_divide_when_given_negative_numbers_returns_correct_result():
    assert divide(5, -2) == -2.5
    assert divide(-2, 5) == -0.4

def test_divide_when_given_zero_denominator_raises_error():
    with pytest.raises(ZeroDivisionError):
        divide(1, 0)
```
</details>


## Create a pull request
Before starting a pull request:
1. Ensure all tests pass: `poe test`
2. Ensure that the code is formatted correctly: `poe format_check`
3. Fix any formatting errors with: `poe format`

Alternatively, you can run 1 & 2 in one command using `poe verify`

Once ready, start a pull request on GitHub and include:
- A clear description of what you're fixing/adding
- Any relevant context or background information

Our pipeline will automatically check your pull request for formatting errors and test results. If you contributed a novel method, it would be great if you can also provide pretrained weights. 
