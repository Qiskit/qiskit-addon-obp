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

"""A wrapper for PauliLindbladError objects as an Instruction."""

from __future__ import annotations

from qiskit.circuit import Instruction
from qiskit_ibm_runtime.utils.noise_learner_result import PauliLindbladError


class PauliLindbladErrorInstruction(Instruction):
    def __init__(self, ple: PauliLindbladError, index: int | None = None):
        self.ple = ple
        self._index = index
        label = "LayerError"
        if index is not None:
            label = f"{label} {index}"
        super().__init__(
            name="LayerError",
            num_qubits=self.ple.num_qubits,
            num_clbits=0,
            params=[],
            label=label,
        )

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, PauliLindbladErrorInstruction)
            and self._index == other._index
            and (self.ple.rates == other.ple.rates).all()
            and (self.ple.generators == other.ple.generators)
        )

    @property
    def index(self):
        if self._index is None:
            raise ValueError("Index not defined, you probably didn't mean to call this.")
        return self._index
