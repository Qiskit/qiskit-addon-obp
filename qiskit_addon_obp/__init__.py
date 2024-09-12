# This code is a Qiskit project.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Main operator backpropagation functionality.

.. currentmodule:: qiskit_addon_obp

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   backpropagate

Submodules
==========

.. autosummary::
   :toctree:

   utils
"""

from .backpropagation import backpropagate

__all__ = [
    "backpropagate",
]
