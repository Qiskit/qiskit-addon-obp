############################################
Qiskit addon: operator backpropagation (OBP)
############################################

`Qiskit addons <https://docs.quantum.ibm.com/guides/addons>`_ are a collection of modular tools for building utility-scale workloads powered by Qiskit.

This package contains the Qiskit addon for operator backpropagation (OBP).
Experimental errors limit the depth of quantum circuits that can be executed
on near-term devices. OBP is a technique to reduce circuit depth by trimming
operations from its end at the cost of more operator measurements.

As one backpropagates an operator further through a circuit, the size of the
observable will grow exponentially, which results in both a classical and
quantum resource overhead. However, for some circuits, the resulting
distribution of Pauli observables is more concentrated than the worst-case
exponential scaling, meaning that some terms in the Hamiltonian with small
coefficients can be truncated to reduce the quantum overhead. The error
incurred by doing this can be controlled to find a suitable tradeoff between
precision and efficiency.

There are a number of ways in which operator backpropagation can be performed,
this package uses a method based on Clifford perturbation theory, which has the
benefit that the overhead incurred by backpropagating various gates is
determined by the non-Cliffordness of that gate. This leads to an increased
efficiency for some families of circuits relative to tensor-network based
methods for OBP, which currently have high classical overheads even in cases
where the quantum overhead remains tame.

Documentation
-------------

All documentation is available `here <https://qiskit.github.io/qiskit-addon-obp/>`_.

Installation
------------

We encourage installing this package via ``pip``, when possible:

.. code-block:: bash

   pip install 'qiskit-addon-obp'


For more installation information refer to the `installation instructions <install.rst>`_ in the documentation.

Deprecation Policy
------------------

We follow `semantic versioning <https://semver.org/>`_ and are guided by the principles in
`Qiskit's deprecation policy <https://github.com/Qiskit/qiskit/blob/main/DEPRECATION.md>`_.
We may occasionally make breaking changes in order to improve the user experience.
When possible, we will keep old interfaces and mark them as deprecated, as long as they can co-exist with the
new ones.
Each substantial improvement, breaking change, or deprecation will be documented in the
`release notes <https://qiskit.github.io/qiskit-addon-obp/release-notes.html>`_.

Contributing
------------

The source code is available `on GitHub <https://github.com/Qiskit/qiskit-addon-obp>`_.

The developer guide is located at `CONTRIBUTING.md <https://github.com/Qiskit/qiskit-addon-obp/blob/main/CONTRIBUTING.md>`_
in the root of this project's repository.
By participating, you are expected to uphold Qiskit's `code of conduct <https://github.com/Qiskit/qiskit/blob/main/CODE_OF_CONDUCT.md>`_.

We use `GitHub issues <https://github.com/Qiskit/qiskit-addon-obp/issues/new/choose>`_ for tracking requests and bugs.

License
-------

`Apache License 2.0 <https://github.com/Qiskit/qiskit-addon-obp/blob/main/LICENSE.txt>`_


.. toctree::
  :hidden:

   Documentation Home <self>
   Installation Instructions <install>
   Tutorials <tutorials/index>
   How-To Guides <how_tos/index>
   API Reference <apidocs/index>
   GitHub <https://github.com/Qiskit/qiskit-addon-obp>
   Release Notes <release-notes>
