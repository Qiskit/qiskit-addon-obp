########################
Operator backpropagation
########################

Operator backpropagation (OBP) is a technique for propagating an observable backward through gates at the end of a quantum circuit. This results in a more shallow circuit at the cost of an increase in observable measurement bases. As one backpropagates an operator further through a circuit, the size of the observable grows exponentially, resulting in both a classical and quantum resource overhead. However, for some circuits, the resulting distribution of Pauli observables is more concentrated than the worst-case exponential scaling, meaning that some terms in the Hamiltonian with small coefficients can be truncated to reduce the quantum overhead. The error incurred by doing this can be controlled to find a suitable tradeoff between precision and efficiency.

There are a number of ways in which operator backpropagation can be performed, this package uses a method based on Clifford perturbation theory, which has the benefit that the overhead incurred by backpropagating various gates is determined by the non-Cliffordness of that gate. This leads to an increased efficiency for some families of circuits relative to tensor-network based methods for OBP, which currently have high classical overheads even in cases where the quantum overhead remains tame.

This package is suitable for estimating expectation values of general quantum circuits and Pauli observables; however, for high-magic (very non-Clifford) circuits the observable error incurred from truncation will likely become prohibitive before the observable has been propagated through a meaningful portion of the circuit. For near-Clifford circuits, it may be possible to propagate through much more of the circuit while maintaining relatively small error bounds.

Getting started
---------------

A simple guide to help you get started quickly with this package is available in the :doc:`quick start guide <guides/quickstart>`.

Workflow examples
-----------------

Trotterized time evolution of a 2D spin model `[1] <https://www.nature.com/articles/s41534-026-01196-0>`_.

Technical discussion
--------------------

- **A single entry point:** the ``backpropagate`` function performs OBP.
- **Two independent controls over the depth ↔ accuracy tradeoff:**

  - ``OperatorBudget`` bounds how large the observable may grow during backpropagation.
  - ``TruncationErrorBudget`` bounds how much error may be incurred during backpropagation.

- **Effectiveness depends on how Clifford the circuit is:** for high-magic circuits (i.e. very non-Clifford), terms added to the observable tend to have larger coefficients, so one incurs more truncation error earlier, limiting the depth savings under a fixed error budget. Near-Clifford circuits can be truncated more aggressively and backpropagated deeper under the same budget.
- **Fully introspectable:** each run returns an ``OBPMetadata`` object recording per-slice Pauli counts, QWC-group counts, and accumulated truncation error, with ``plot_*`` helpers to visualize operator growth and error against circuit depth — useful for tuning budgets before committing QPU time.

Contributing
------------

The developer guide is located at `CONTRIBUTING.md <https://github.com/Qiskit/qiskit-addon-obp/blob/main/CONTRIBUTING.md>`_
in the root of this project's repository.
By participating, you are expected to uphold Qiskit's `code of conduct <https://github.com/Qiskit/qiskit/blob/main/CODE_OF_CONDUCT.md>`_.

We use `GitHub issues <https://github.com/Qiskit/qiskit-addon-obp/issues/new/choose>`_ for tracking requests and bugs.

License
-------

`Apache License 2.0 <https://github.com/Qiskit/qiskit-addon-obp/blob/main/LICENSE.txt>`_

Deprecation policy
------------------

This package follows `semantic versioning <https://semver.org/>`_. We may occasionally make breaking changes in order to improve
the user experience. When possible, we will keep old interfaces and mark them as deprecated, as long as they can co-exist with the new ones.
Each substantial improvement, breaking change, or deprecation will be documented in the
`release notes <https://qiskit.github.io/qiskit-addon-obp/release-notes.html>`_.

References
----------

1. B. Fuller *et al.*, "`Improved quantum computation using operator backpropagation <https://www.nature.com/articles/s41534-026-01196-0>`_," *npj Quantum Inf.* **12**, 51 (2026). [`arXiv <https://arxiv.org/abs/2502.01897>`_]

.. toctree::
   :hidden:

   Documentation home <self>
   Installation instructions <install>
   Guides <guides/index>
   GitHub <https://github.com/Qiskit/qiskit-addon-obp>

.. toctree::
   :hidden:
   :caption: Tutorials

   Operator backpropagation (OBP) for estimation of expectation values <https://quantum.cloud.ibm.com/docs/en/tutorials/operator-back-propagation>

.. toctree::
   :hidden:
   :caption: API reference

   Python API reference <https://quantum.cloud.ibm.com/docs/en/api/qiskit-addon-obp>
   Release notes <release-notes>
