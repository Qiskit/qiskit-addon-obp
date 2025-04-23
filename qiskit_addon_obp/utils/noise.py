# This code is a Qiskit project.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Utilities for noise operators."""

from __future__ import annotations

from qiskit.circuit import Instruction
from qiskit_ibm_runtime.utils.noise_learner_result import PauliLindbladError


class PauliLindbladErrorInstruction(Instruction):
    """A lightweight wrapper around a ``PauliLindbladError``."""

    def __init__(self, ple: PauliLindbladError):
        """Initializes a circuit instruction with a Pauli-Lindblad error.

        Args:
            ple: the Pauli-Lindblad error to include.
            index: an optional index of this error instruction.
        """
        self._ple = ple
        label = "LayerError"
        super().__init__(
            name="LayerError",
            num_qubits=self._ple.num_qubits,
            num_clbits=0,
            params=[],
            label=label,
        )

    @property
    def ple(self) -> PauliLindbladError:
        """Returns the internal Pauli-Lindblad error object."""
        return self._ple
