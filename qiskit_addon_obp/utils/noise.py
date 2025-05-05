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

import numpy as np
from qiskit.circuit import Instruction, QuantumCircuit, QuantumRegister
from qiskit.quantum_info import PauliList, SparsePauliOp, SuperOp
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

    def _define(self) -> None:
        # Implementation of Instruction._define which populates the definition attribute with a
        # QuantumCircuit-based representation of this instruction.

        # This implementation is adapted from qiskit_aer.noise.PauliLindbladError
        chan_z = np.zeros((1, self.num_qubits), dtype=bool)
        chan_x = np.zeros_like(chan_z)
        chan_p = np.ones(1, dtype=float)
        for term_z, term_x, term_r in zip(
            self.ple.generators.z,
            self.ple.generators.x,
            self.ple.rates,
        ):
            term_p = 0.5 - 0.5 * np.exp(-2 * term_r)
            chan_z = np.concatenate([chan_z, np.logical_xor(chan_z, term_z)], axis=0)
            chan_x = np.concatenate([chan_x, chan_x ^ term_x])
            chan_p = np.concatenate([(1 - term_p) * chan_p, term_p * chan_p])

        error_op = SparsePauliOp(PauliList.from_symplectic(chan_z, chan_x), chan_p).simplify()
        chan = SuperOp(np.zeros(2 * [4**self.num_qubits]))
        for pauli, coeff in zip(error_op.paulis, error_op.coeffs.real):
            chan += coeff * SuperOp(pauli)

        q = QuantumRegister(self.num_qubits, "q")
        qc = QuantumCircuit(q, name=self.name)
        qc.append(chan.to_instruction(), q)

        self.definition = qc
