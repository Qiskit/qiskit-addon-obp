from dataclasses import asdict
from typing import Callable

import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorEstimator
from qiskit.primitives.containers import PrimitiveResult
from qiskit.primitives.containers.data_bin import DataBin
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit.primitives.containers.pub_result import PubResult
from qiskit.quantum_info import Pauli, SparsePauliOp, StabilizerState

from qiskit_addon_obp import backpropagate
from qiskit_addon_obp.utils.metadata import OBPMetadata
from qiskit_addon_obp.utils.simplify import OperatorBudget
from qiskit_addon_obp.utils.truncating import TruncationErrorBudget

from .utils.broadcast_indices_iter import broadcast_indicies_iter


class OBPEstimator(StatevectorEstimator):
    # This is only done to avoid repeating of some of the abstract methods required
    # by the BaseEstimatorV2 class. This code does not call the AerSimulator or
    # StateVector simulator at all.

    def __init__(
        self,
        slicer: Callable | None = None,
        operator_budget: OperatorBudget | None = None,
        truncation_error_budget: TruncationErrorBudget | None = None,
        max_seconds: int | None = None,
    ):
        if slicer is None:

            def _slicer(qc):
                return [qc]

            self._slicer = _slicer
        else:
            self._slicer = slicer

        self._operator_budget = operator_budget
        self._truncation_error_budget = truncation_error_budget
        self._max_seconds = max_seconds
        super().__init__()

    def _run(self, pubs: list[EstimatorPub]) -> PrimitiveResult[PubResult]:
        return super()._run(list(pubs))

    def _run_pub(self, pub: EstimatorPub) -> PubResult:
        shape = pub.shape if pub.shape else (1,)
        evs = np.ones(shape, dtype=complex) * np.nan
        obp_metadata_array = np.ones(shape, dtype=object) * np.nan

        inds = broadcast_indicies_iter(
            pub.observables.shape,
            pub.parameter_values.shape,
        )

        _par_idx_prev = None
        for obs_idx, par_idx, ev_idx in inds:
            if _par_idx_prev != par_idx:
                bound_circ = pub.parameter_values.bind(
                    circuit=pub.circuit,
                    loc=par_idx,
                )
                bound_qc_slices = self._slicer(bound_circ)

            _ev, _obp_metadata = self._run_pub_idx(
                qc=bound_circ,
                obs=pub.observables[obs_idx],
                pre_sliced_qc=bound_qc_slices,
            )

            evs[ev_idx] = _ev
            obp_metadata_array[ev_idx] = asdict(_obp_metadata)

        pub_res = PubResult(
            data=DataBin(shape=evs.shape, evs=evs, obp_metadata=obp_metadata_array)
        )
        pub_res.metadata["circuit_metadata"] = pub.circuit.metadata.copy()

        return pub_res

    def _run_pub_idx(
        self,
        qc: QuantumCircuit,
        obs: SparsePauliOp,
        pre_sliced_qc: list[QuantumCircuit] | None = None,
    ) -> tuple[float | complex, OBPMetadata]:
        new_obs_list, new_qc, obp_metadata = backpropagate(
            observables=[SparsePauliOp.from_list([(_p, _c) for _p, _c in obs.items()])],
            slices=self._slicer(qc=qc) if pre_sliced_qc is None else pre_sliced_qc,
            operator_budget=self._operator_budget,
        )
        new_obs: SparsePauliOp = new_obs_list[0]

        if new_qc:
            # In this case, one alternative is to have the user supply an auxiliary estimator
            # The auxiliary estimator would compute the expectation value wrt the remaining
            # slices. E.g. The primary use case would likely be providing an estimator which
            # uses an actual backend to compute the remaining expectation value, etc.
            raise NotImplementedError(
                "Observable did not push through circuit entirely, this is not yet supported"
            )

        _stab_state = StabilizerState(QuantumCircuit(qc.num_qubits))
        _ev = 0
        for _ps, _c in new_obs.to_list():
            _ev += _c * _stab_state.expectation_value(Pauli(_ps))

        return _ev, obp_metadata
