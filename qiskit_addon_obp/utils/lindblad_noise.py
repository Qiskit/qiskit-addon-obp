from collections.abc import Sequence
from typing import Optional

import numpy as np
from qiskit.circuit import Instruction, QuantumRegister, Qubit
from qiskit.quantum_info import Pauli, PauliList
from qiskit_ibm_runtime.utils.noise_learner_result import PauliLindbladError


class PauliLindbladErrorInstruction(Instruction):
    def __init__(self, ple: PauliLindbladError, index: Optional[int] = None):
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
        if isinstance(other, PauliLindbladErrorInstruction):
            if self._index != other._index:
                return False
            if (self.ple.rates != other.ple.rates).any():
                return False
            if self.ple.generators != other.ple.generators:
                return False
            return True
        return False

    @property
    def index(self):
        if self._index is None:
            raise ValueError("Index not defined, you probably didn't mean to call this.")
        return self._index

    def _define(self):
        self._definition = pauli_lindblad_error_to_qc(self.ple, instr=False)

    def map_to_inds(
        self, qargs: Sequence[Qubit], target_qreg: QuantumRegister
    ) -> "PauliLindbladErrorInstruction":
        qarg_map = {v: k for k, v in enumerate(list(target_qreg))}
        qarg_inds = [qarg_map[qubit] for qubit in qargs]

        x_ = np.zeros((len(self.ple.generators), len(target_qreg)))
        z_ = np.zeros((len(self.ple.generators), len(target_qreg)))

        x_[:, qarg_inds] = self.ple.generators.x
        z_[:, qarg_inds] = self.ple.generators.z

        return PauliLindbladErrorInstruction(
            ple=PauliLindbladError(
                generators=PauliList.from_symplectic(z_, x_, 0),
                rates=self.ple.rates,
            ),
            index=self.index,
        )


def evolve_pauli_lindblad_error_instruction(
    pauli: Pauli,
    ple: PauliLindbladError,
    ple_qargs_in_op: list[int],
) -> tuple[float, np.ndarray]:
    fid = 1.0
    for gen, rate in zip(ple.generators, ple.rates):
        prob_no_err = (1 + np.exp(-2 * rate)) / 2
        prob = 1 - prob_no_err

        gen = gen.apply_layout(ple_qargs_in_op)

        if not pauli.anticommutes(gen):
            fid *= 1.0
        else:
            fid *= 1 - 2.0 * prob

    return fid
