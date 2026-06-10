############################################
Operator backpropagation (OBP)
############################################

This package contains the Qiskit addon for operator backpropagation (OBP).
Experimental errors limit the depth of quantum circuits that can be executed
on near-term devices. OBP is a technique to reduce circuit depth by trimming
operations from its end, at the cost of more operator measurements.

As one backpropagates an operator further through a circuit, the size of the
observable will grow exponentially, which results in both a classical and
quantum resource overhead. However, for some circuits, the resulting
distribution of Pauli observables is more concentrated than the worst-case
exponential scaling, meaning that some terms in the Hamiltonian with small
coefficients can be truncated to reduce the quantum overhead. The error
incurred by doing this can be controlled to find a suitable tradeoff between
precision and efficiency.

There are several ways to perform operator backpropagation. 
This package uses a method based on Clifford perturbation theory, which has the
benefit that the overhead incurred by backpropagating gates is
determined by the non-Cliffordness of that gate. This leads to an increased
efficiency for some families of circuits relative to tensor-network based
methods for OBP, which currently have high classical overheads even in cases
where the quantum overhead remains low.

Deprecation policy
------------------

This package follows `semantic versioning <https://semver.org/>`_ and is guided by the principles in
`Qiskit's deprecation policy <https://github.com/Qiskit/qiskit/blob/main/DEPRECATION.md>`_.
We may occasionally make breaking changes in order to improve the user experience.
When possible, we will keep old interfaces and mark them as deprecated, as long as they can co-exist with the
new ones.
Each substantial improvement, breaking change, or deprecation will be documented in the
`release notes <https://qiskit.github.io/qiskit-addon-obp/release-notes.html>`_.

Contributing
------------

The developer guide is located at `CONTRIBUTING.md <https://github.com/Qiskit/qiskit-addon-obp/blob/main/CONTRIBUTING.md>`_
in the root of this project's repository.
By participating, you are expected to uphold Qiskit's `code of conduct <https://github.com/Qiskit/qiskit/blob/main/CODE_OF_CONDUCT.md>`_.

We use `GitHub issues <https://github.com/Qiskit/qiskit-addon-obp/issues/new/choose>`_ for tracking requests and bugs.

References
----------

1. B. Fuller et al. `Improved Quantum Computation using Operator Backpropagation <https://arxiv.org/abs/2502.01897>`_, arXiv:2502.01897 [quant-ph].

License
-------

`Apache License 2.0 <https://github.com/Qiskit/qiskit-addon-obp/blob/main/LICENSE.txt>`_


.. toctree::
  :hidden:

   Documentation Home <self>
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

   Python API reference <apidocs/index>
   Release notes <release-notes>
