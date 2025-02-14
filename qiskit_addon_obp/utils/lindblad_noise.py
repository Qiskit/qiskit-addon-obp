from typing import Optional, Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Instruction, QuantumRegister, Qubit
from qiskit.quantum_info import Pauli, PauliList, SparsePauliOp
from qiskit_aer.noise import pauli_error
from qiskit_ibm_runtime.utils.noise_learner_result import PauliLindbladError


class PauliLindbladErrorInstruction(Instruction):
    def __init__(self, ple: PauliLindbladError, index: Optional[int] = None):
        self._ple = ple
        self._index = index
        label = "LayerError"
        if index is not None:
            label = f"{label} {index}"
        super().__init__(
            name="LayerError",
            num_qubits=self._ple.num_qubits,
            num_clbits=0,
            params=[],
            label=label,
        )

    def __eq__(self, other) -> bool:
        if isinstance(other, PauliLindbladErrorInstruction):
            if self._index != other._index:
                return False
            if (self._ple.rates != other._ple.rates).any():
                return False
            if self._ple.generators != other._ple.generators:
                return False
            return True
        return False

    @property
    def index(self):
        if self._index is None:
            raise ValueError(
                "Index not defined, you probably didn't mean to call this."
            )
        return self._index

    def _define(self):
        self._definition = pauli_lindblad_error_to_qc(self._ple, instr=False)

    def map_to_inds(
        self, qargs: Sequence[Qubit], target_qreg: QuantumRegister
    ) -> "PauliLindbladErrorInstruction":
        qarg_map = {v: k for k, v in enumerate(list(target_qreg))}
        qarg_inds = [qarg_map[qubit] for qubit in qargs]

        x_ = np.zeros((len(self._ple.generators), len(target_qreg)))
        z_ = np.zeros((len(self._ple.generators), len(target_qreg)))

        x_[:, qarg_inds] = self._ple.generators.x
        z_[:, qarg_inds] = self._ple.generators.z

        return PauliLindbladErrorInstruction(
            ple=PauliLindbladError(
                generators=PauliList.from_symplectic(z_, x_, 0),
                rates=self._ple.rates,
            ),
            index=self.index,
        )


class _PauliEvolutionData:
    def __init__(
        self,
        gen: Pauli,
        prop_pauli: Pauli,
        prob: Optional[float] = None,
        rate: Optional[float] = None,
    ):
        if prob is None and rate is None:
            raise ValueError("must have either prob or rate defined")
        if (prob is not None) and (rate is not None):
            raise ValueError("cannot provide both rate and prob")

        if prob is None:
            prob_no_err = (1 + np.exp(-2 * rate)) / 2
            prob = 1 - prob_no_err
        if rate is None:
            prob_no_err = 1 - prob
            rate = -1 / 2 * np.log(2 * prob_no_err - 1)

        self.rate = rate
        self.prob = prob

        self.gen = gen
        self.prop_pauli = prop_pauli

        self._ac = self.prop_pauli.anticommutes(self.gen)

    @property
    def fid(self):
        if not self.anti_comm:
            return 1.0
        return 1 - 2.0 * self.prob

    @property
    def anti_comm(self):
        return self._ac

    @property
    def comm(self):
        return not self.anti_comm


def evolve_pauli_lindblad_error_instruction(
    pauli: Pauli,
    ple_instr: PauliLindbladErrorInstruction,
) -> tuple[float, np.ndarray]:
    fid = 1.0
    for gen, rate in zip(ple_instr._ple.generators, ple_instr._ple.rates):
        _ped = _PauliEvolutionData(rate=rate, gen=gen, prop_pauli=pauli)
        fid *= _ped.fid
    return fid


def pauli_lindblad_error_to_qc(ple: PauliLindbladError, instr: bool = True):
    qc_res = QuantumCircuit(ple.num_qubits)
    p_id = Pauli("I" * ple.num_qubits)
    for pauli, rate in zip(ple.generators, ple.rates):
        if pauli == p_id:
            continue
        prob_no_err = (1 + np.exp(-2 * rate)) / 2
        err = pauli_error([(pauli, 1 - prob_no_err), (p_id, prob_no_err)])
        qc_res.append(err, qargs=range(qc_res.num_qubits))
    if instr:
        return qc_res.to_instruction(label=ple.label)
    return qc_res


def evolve_ple_spo(
    spo: SparsePauliOp,
    ple_instr: PauliLindbladErrorInstruction,
):
    coeffs_new = []
    for pauli, coeff in zip(spo.paulis, spo.coeffs):
        _coeff = coeff
        _coeff *= evolve_pauli_lindblad_error_instruction(pauli, ple_instr)
        coeffs_new.append(_coeff)
    return SparsePauliOp(spo.paulis, coeffs_new)
