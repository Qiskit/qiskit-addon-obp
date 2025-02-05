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

# Qiskit addon: operator backpropagation (OBP)

### Table of contents

* [About](#about)
* [Documentation](#documentation)
* [Installation](#installation)
* [Deprecation Policy](#deprecation-policy)
* [Contributing](#contributing)
* [License](#license)

----------------------------------------------------------------------------------------------------

### About

[Qiskit addons](https://docs.quantum.ibm.com/guides/addons) are a collection of modular tools for building utility-scale workloads powered by Qiskit.

This package contains the Qiskit addon for operator backpropagation (OBP). Experimental errors limit the depth of quantum circuits that can be executed on near-term devices. OBP is a technique to reduce circuit depth by trimming operations from its end at the cost of more operator measurements.

As one backpropagates an operator further through a circuit, the size of the observable will grow exponentially, which results in both a classical and quantum resource overhead. However, for some circuits, the resulting distribution of Pauli observables is more concentrated than the worst-case exponential scaling, meaning that some terms in the Hamiltonian with small coefficients can be truncated to reduce the quantum overhead. The error incurred by doing this can be controlled to find a suitable tradeoff between precision and efficiency. 

There are a number of ways in which operator backpropagation can be performed, this package uses a method based on Clifford perturbation theory, which has the benefit that the overhead incurred by backpropagating various gates is determined by the non-Cliffordness of that gate. This leads to an increased efficiency for some families of circuits relative to tensor-network based methods for OBP, which currently have high classical overheads even in cases where the quantum overhead remains tame. 

----------------------------------------------------------------------------------------------------

### Documentation

All documentation is available at https://qiskit.github.io/qiskit-addon-obp/.

----------------------------------------------------------------------------------------------------

### Installation

We encourage installing this package via `pip`, when possible:

```bash
pip install 'qiskit-addon-obp'
```

For more installation information refer to these [installation instructions](docs/install.rst).

----------------------------------------------------------------------------------------------------

### Deprecation Policy

We follow [semantic versioning](https://semver.org/) and are guided by the principles in
[Qiskit's deprecation policy](https://github.com/Qiskit/qiskit/blob/main/DEPRECATION.md).
We may occasionally make breaking changes in order to improve the user experience.
When possible, we will keep old interfaces and mark them as deprecated, as long as they can co-exist with the
new ones.
Each substantial improvement, breaking change, or deprecation will be documented in the
[release notes](https://qiskit.github.io/qiskit-addon-obp/release-notes.html).

----------------------------------------------------------------------------------------------------

### Contributing

The source code is available [on GitHub](https://github.com/Qiskit/qiskit-addon-obp).

The developer guide is located at [CONTRIBUTING.md](https://github.com/Qiskit/qiskit-addon-obp/blob/main/CONTRIBUTING.md)
in the root of this project's repository.
By participating, you are expected to uphold Qiskit's [code of conduct](https://github.com/Qiskit/qiskit/blob/main/CODE_OF_CONDUCT.md).

We use [GitHub issues](https://github.com/Qiskit/qiskit-addon-obp/issues/new/choose) for tracking requests and bugs.

----------------------------------------------------------------------------------------------------

### References

1. B. Fuller et al. [Improved Quantum Computation using Operator Backpropagation](https://arxiv.org/abs/2502.01897), arXiv:2502.01897 [quant-ph]. 

----------------------------------------------------------------------------------------------------

### License

[Apache License 2.0](LICENSE.txt)
