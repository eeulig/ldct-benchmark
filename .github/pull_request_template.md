<!-- Please complete this template entirely. -->


Changes proposed in this pull request:
<!-- Please list all changes/additions here. -->
-

<!-- Please complete the following checklist! Only leave the relevant subsection(s) based on what your PR implements. -->
### Checklist
<!-- Please only keep the relevant subsection 
Documentation: If you've added or updated documentation.
Fix: If you've fixed a bug or issue.
Feature: If you've added a new feature.
Denoising Method: If you've added a new LDCT denoising method.
-->
- [ ] I've read and followed all steps in the [contributing guide](https://github.com/eeulig/ldct-benchmark/blob/main/CONTRIBUTING.md).

#### Documentation
- [ ] I've checked that the docs build correctly locally by running `mkdocs serve`.

#### Fix
- [ ] I've added unit tests and gave them meaningful names. Ideally, I added a test that fails without my fix and passes with it.
- [ ] I've updated or added meaningful docstrings in [numpy format](https://numpydoc.readthedocs.io/en/latest/format.html).
- [ ] I ran `poe verify` and checked that all tests pass.

#### Feature
- [ ] I've added unit tests and gave them meaningful names.
- [ ] I've updated or added meaningful docstrings in [numpy format](https://numpydoc.readthedocs.io/en/latest/format.html).
- [ ] I ran `poe verify` and checked that all tests pass.

#### Denoising Method
- [ ] I've added unit tests and gave them meaningful names.
- [ ] I've updated or added meaningful docstrings in [numpy format](https://numpydoc.readthedocs.io/en/latest/format.html). The docstring of the main trainer class contains a reference to the original publication.
- [ ] I ran `poe verify` and checked that all tests pass.
- [ ] I've added the method to the [table of implemented algorithms](https://github.com/eeulig/ldct-benchmark/blob/main/docs/denoising_algorithms.md#implemented-algorithms) **including a reference to the original publication**.
- [ ] I've evaluated my algorithm and reported its performance [here](https://github.com/eeulig/ldct-benchmark/blob/main/docs/denoising_algorithms.md#test-set-performance).
- [ ] I would like to contribute weights for the trained model to the model hub.