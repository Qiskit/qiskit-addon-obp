Installation instructions
=========================

Before installing the package, choose how you're going to run and install the
packages. There are two primary ways to do this:

- :ref:`Option 1`
- :ref:`Option 2`

Preinstallation
^^^^^^^^^^^^^^^^

First, create a minimal environment with only Python installed in it. We recommend using `Python virtual environments <https://docs.python.org/3.10/tutorial/venv.html>`__.

.. code:: sh

    python3 -m venv /path/to/virtual/environment

Activate your new environment.

.. code:: sh

    source /path/to/virtual/environment/bin/activate

Note: If you are using Windows, use the following commands in PowerShell:

.. code:: pwsh

    python3 -m venv c:\path\to\virtual\environment
    c:\path\to\virtual\environment\Scripts\Activate.ps1


.. _Option 1:

Option 1: Install from PyPI
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The most straightforward way to install the ``qiskit-addon-obp`` package is by using ``PyPI``.

.. code:: sh

    pip install 'qiskit-addon-obp'


.. _Option 2:

Option 2: Install from source
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Users who want to develop in the repository or run the notebooks locally might want to install from source.

If so, the first step is to clone the ``qiskit-addon-obp`` repository.

.. code:: sh

    git clone git@github.com:Qiskit/qiskit-addon-obp.git

Next, upgrade pip and enter the repository.

.. code:: sh

    pip install --upgrade pip
    cd qiskit-addon-obp

The next step is to install ``qiskit-addon-obp`` to the virtual environment. If you plan on running the notebooks, install the
notebook dependencies so you can run visualizations. If you plan on developing in the repository, you
might want to install the ``dev`` dependencies.

Adjust the options below to suit your needs.

.. code:: sh

    pip install tox notebook -e '.[notebook-dependencies,dev]'

If you installed the notebook dependencies, you can get started by running the notebooks in the docs directory.

.. code::

    cd docs/
    jupyter lab
