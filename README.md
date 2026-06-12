<!-- SHIELDS -->
<div align="left">

  [![Release](https://img.shields.io/pypi/v/qiskit-addon-obp.svg?label=Release)](https://github.com/Qiskit/qiskit-addon-obp/releases)
  ![Platform](https://img.shields.io/badge/%F0%9F%92%BB%20Platform-Linux%20%7C%20macOS%20%7C%20Windows-informational)
  [![Python](https://img.shields.io/pypi/pyversions/qiskit-addon-obp?label=Python&logo=python)](https://www.python.org/)
  [![Qiskit](https://img.shields.io/badge/Qiskit%20-%20%3E%3D1.2%20-%20%236133BD?logo=Qiskit)](https://github.com/Qiskit/qiskit)
<br />
  [![Docs (stable)](https://img.shields.io/badge/%F0%9F%93%84%20Docs-stable-blue.svg)](https://qiskit.github.io/qiskit-addon-obp/)
  <!--[![DOI](https://zenodo.org/badge/TODO.svg)](https://zenodo.org/badge/latestdoi/TODO)-->
  [![License](https://img.shields.io/github/license/Qiskit/qiskit-addon-obp?label=License)](LICENSE.txt)
  [![Downloads](https://img.shields.io/pypi/dm/qiskit-addon-obp.svg?label=Downloads)](https://pypi.org/project/qiskit-addon-obp/)
  [![Tests](https://github.com/Qiskit/qiskit-addon-obp/actions/workflows/test_latest_versions.yml/badge.svg)](https://github.com/Qiskit/qiskit-addon-obp/actions/workflows/test_latest_versions.yml)
  [![Coverage](https://coveralls.io/repos/github/Qiskit/qiskit-addon-obp/badge.svg?branch=main)](https://coveralls.io/github/Qiskit/qiskit-addon-obp?branch=main)

# Operator backpropagation

This package implements the operator backpropagation technique for reducing depth of circuits in expectation value calculations. Operator backpropagation (OBP) is a technique for propagating an observable backward through gates at the end of a quantum circuit. This results in a more shallow circuit at the cost of an increase in observable measurement bases. As one backpropagates an operator further through a circuit, the size of the observable grows exponentially, resulting in both a classical and quantum resource overhead. However, for some circuits, the resulting distribution of Pauli observables is more concentrated than the worst-case exponential scaling, meaning that some terms in the Hamiltonian with small coefficients can be truncated to reduce the quantum overhead. The error incurred by doing this can be controlled to find a suitable tradeoff between precision and efficiency.

There are a number of ways in which operator backpropagation can be performed, this package uses a method based on Clifford perturbation theory, which has the benefit that the overhead incurred by backpropagating various gates is determined by the non-Cliffordness of that gate. This leads to an increased efficiency for some families of circuits relative to tensor-network based methods for OBP, which currently have high classical overheads even in cases where the quantum overhead remains tame.

This package is suitable for estimating expectation values of general quantum circuits and Pauli observables; however, for high-magic (very non-Clifford) circuits the observable error incurred from truncation will likely become prohibitive before the observable has been propagated through a meaningful portion of the circuit. For near-Clifford circuits, it may be possible to propagate through much more of the circuit while maintaining relatively small error bounds.

----------------------------------------------------------------------------------------------------

### Documentation

[Documentation](https://quantum.cloud.ibm.com/docs/addons/qiskit-addon-obp) for this package is located on the IBM Quantum Platform.

----------------------------------------------------------------------------------------------------

### Installation

We encourage installing this package via `pip`, when possible:

```bash
pip install 'qiskit-addon-obp'
```

For more installation information refer to these [installation instructions](docs/install.rst).

----------------------------------------------------------------------------------------------------

### Getting started

A simple guide to help you get started quickly with this package is available [here](docs/guides/quickstart.ipynb).

----------------------------------------------------------------------------------------------------

### Use case examples

This technique has been used to implement lower-depth Trotter circuits for the time-evolution of a 2D spin model [[1]](https://www.nature.com/articles/s41534-026-01196-0).

----------------------------------------------------------------------------------------------------

### Technical discussion

- **A single function for performing OBP, `backpropagate`**
- **Two independent controls over the depth ↔ accuracy tradeoff**
    - `OperatorBudget` sets bounds on how large the observable may grow during backpropagation
    - `TruncationErrorBudget` sets bounds on how much error can be incurred during backpropagation
- **Effectiveness depends on how Clifford the circuit is:** For high-magic circuits (i.e. very non-Clifford), terms added to the observable during backpropagation tend to have larger coefficients, which means one will generally incur a large amount of truncation error earlier during backpropagation, limiting the depth savings under a fixed error budget. For circuits that are near-Clifford, one can generally truncate terms more aggressively and backpropagate deeper into the circuit while staying under a fixed error budget compared to high-magic circuits.
- **Fully introspectable:** Each run returns an `OBPMetadata` object recording per-slice Pauli counts, QWC-group counts, and accumulated truncation error, with `plot_*` helpers to visualize operator growth and error against circuit
depth — useful for tuning budgets before committing QPU time.

----------------------------------------------------------------------------------------------------

### Contributing

The source code is available [on GitHub](https://github.com/Qiskit/qiskit-addon-obp).

The developer guide is located at [CONTRIBUTING.md](https://github.com/Qiskit/qiskit-addon-obp/blob/main/CONTRIBUTING.md)
in the root of this project's repository.
By participating, you are expected to uphold Qiskit's [code of conduct](https://github.com/Qiskit/qiskit/blob/main/CODE_OF_CONDUCT.md).

We use [GitHub issues](https://github.com/Qiskit/qiskit-addon-obp/issues/new/choose) for tracking requests and bugs.

----------------------------------------------------------------------------------------------------

### Citing this package

If you use this package in your research, use the [CITATION.bib](CITATION.bib) file in this project’s repository to cite the appropriate reference(s).

----------------------------------------------------------------------------------------------------

### License

[Apache License 2.0](LICENSE.txt)

----------------------------------------------------------------------------------------------------

### Deprecation policy

This project follows [semantic versioning](https://semver.org/). We may occasionally make breaking changes in order
to improve the user experience. When possible, we will keep old interfaces and mark them as deprecated, as long as
they can co-exist with the new ones. Each substantial improvement, breaking change, or deprecation will be documented
in the [release notes](https://qiskit.github.io/qiskit-addon-obp/release-notes.html).

----------------------------------------------------------------------------------------------------

### References

1. B. Fuller *et al.*, "[Improved quantum computation using operator backpropagation](https://www.nature.com/articles/s41534-026-01196-0)," *npj Quantum Inf.* **12**, 51 (2026). [[arXiv](https://arxiv.org/abs/2502.01897)]
