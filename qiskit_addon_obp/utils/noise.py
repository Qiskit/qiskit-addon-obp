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

    def __init__(self, ple: PauliLindbladError, index: int | None = None):
        """Initializes a circuit instruction with a Pauli-Lindblad error.

        Args:
            ple: the Pauli-Lindblad error to include.
            index: an optional index of this error instruction.
        """
        self._ple = ple
        self._index = index
        label = "LayerError"
        if index is not None:
            label = f"{label} {index}"  # pragma: no cover
        super().__init__(
            name="LayerError",
            num_qubits=self._ple.num_qubits,
            num_clbits=0,
            params=[],
            label=label,
        )

    def __eq__(self, other) -> bool:
        """Checks the equality of two ``PauliLindbladErrorInstruction`` instances."""
        return (  # pragma: no cover
            isinstance(other, PauliLindbladErrorInstruction)
            and self._index == other._index
            and (self._ple.rates == other._ple.rates).all()
            and (self._ple.generators == other._ple.generators)
        )

    @property
    def index(self) -> int:
        """Returns the index of this Pauli-Lindblad error instruction.

        Raises:
            ValueError: if this instruction has not index.
        """
        if self._index is None:  # pragma: no cover
            raise ValueError(  # pragma: no cover
                "Index not defined, you probably didn't mean to call this."
            )
        return self._index  # pragma: no cover

    @property
    def ple(self) -> PauliLindbladError:
        """Returns the internal Pauli-Lindblad error object."""
        return self._ple
