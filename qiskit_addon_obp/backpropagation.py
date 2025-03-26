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

"""Functionality for backpropagating operators onto quantum circuits."""

from __future__ import annotations

import copy
import logging
import signal
import sys
from collections.abc import Sequence

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import Operator, SparsePauliOp

from .utils.metadata import OBPMetadata, SliceMetadata
from .utils.operations import (
    apply_op_to,
    apply_reset_to,
    reduce_op,
    to_global_op,
)
from .utils.simplify import OperatorBudget
from .utils.simplify import simplify as simplify_sparse_pauli_op
from .utils.truncating import TruncationErrorBudget, truncate_binary_search

LOGGER = logging.getLogger(__name__)


class TimeoutException(Exception):
    """A custom timeout exception used by the ``max_seconds`` option."""

    pass


def backpropagate(
    observables: SparsePauliOp | list[SparsePauliOp],
    slices: Sequence[QuantumCircuit],
    *,
    truncation_error_budget: TruncationErrorBudget | None = None,
    operator_budget: OperatorBudget | None = None,
    max_seconds: int | None = None,
) -> tuple[list[SparsePauliOp], Sequence[QuantumCircuit], OBPMetadata]:
    """Backpropagate slices of quantum circuit operations onto the provided observables.

    This function takes a (list of) observable(s) and backpropagates the provided quantum circuit
    slices **in reverse order** onto the observable(s) until one of the stopping criteria is reached.

    The stopping criteria are values which constrain how large the observable may grow during
    backpropagation. These may be specified via the ``operator_budget`` optional argument. Refer to
    the :class:`~qiskit_addon_obp.utils.simplify.OperatorBudget` class for more details.

    During backpropagation, users may truncate low-weight terms from the output observables,
    potentially reducing the number of experiments needed to run on the QPU. Truncating observable
    terms results in expectation value error proportional to the weights of the truncated terms.
    Users may specify how aggressively to truncate by providing a :class:`.TruncationErrorBudget` to
    the ``truncation_error_budget`` argument. Refer to the
    :func:`~qiskit_addon_obp.utils.truncating.setup_budget` documentation for more details.

    .. warning::
       The ``max_seconds`` argument is not available on Windows!

    Args:
        observables: The observable(s) onto which the circuit is backpropagated.
        slices: A sequence of ``QuantumCircuit`` objects representing a single circuit which
            has been separated into partitions spanning all qubits. **These "slices" will be
            backpropagated in reverse order.** Each slice must span all qubits. One may use the
            tools provided in :external:mod:`qiskit_addon_utils.slicing` to slice a single
            :external:class:`~qiskit.circuit.QuantumCircuit`.
        truncation_error_budget: The error budget used for truncating Pauli terms. Refer to the
            `how-to guide <https://qiskit.github.io/qiskit-addon-obp/how_tos/truncate_operator_terms.html>`__ for a detailed discussion on
            truncating terms from the output operator and bounding the incurred error.
        operator_budget: Constraints on how large the operator may grow during backpropagation. If
            ``None``, a default instance of :class:`~qiskit_addon_obp.utils.simplify.OperatorBudget`
            will be used, and no constraints will be placed on the output operator size.
        max_seconds: The maximum number of seconds to run the backpropagation. If this timeout is
            triggered before the function returns, the metadata of that moment will be returned.
            Note, that this metadata may contain only partial information for the last slice being
            backpropagated.

    Returns:
        - The backpropagated observables.
        - The slices which were not absorbed into the observable during backpropagation.
        - A metadata container.

    Raises:
        RuntimeError: If the ``max_seconds`` argument is attempted to be used on Windows.
        ValueError: All observables and slices must act on equivalent numbers of qubits.
        ValueError: An input observable is larger than the constraints specified by ``operator_budget``.
        ValueError: ``operator_budget.max_paulis`` or ``operator_budget.max_qwc_groups`` is less than 1.

    """
    operator_budget = operator_budget or OperatorBudget()
    truncation_error_budget = truncation_error_budget or TruncationErrorBudget()

    _validate_input_options(
        observables,
        slices,
        operator_budget,
    )

    observable_list, qargs_list = _get_observable_and_qargs_lists(observables)

    # Copy the input data structures to avoid modifying them.
    observables_out = observables_tmp = copy.deepcopy(observable_list)
    qargs_out = qargs_tmp = copy.deepcopy(qargs_list)

    # Prepare a metadata container
    num_observables = len(observable_list)
    metadata = OBPMetadata(
        truncation_error_budget=truncation_error_budget,
        operator_budget=operator_budget,
        num_slices=len(slices),
        backpropagation_history=[],
        num_backpropagated_slices=0,
    )

    if max_seconds is not None:
        if sys.platform == "win32":
            # NOTE: we do cover the line below in our test suite, but our coverage summary only
            # gathers data from UNIX runners.
            raise RuntimeError(  # pragma: no cover
                "The `max_seconds` argument is not available on Windows."
            )

        def handle_timeout(signum, frame):
            raise TimeoutException()

        signal.signal(signal.SIGALRM, handle_timeout)
        signal.alarm(max_seconds)

    try:
        # Iterate over the slices in reverse because we start backpropagating from the end.
        for slice_ in slices[::-1]:
            # Set up a new per-slice metadata container
            slice_metadata = SliceMetadata(
                slice_errors=[0.0] * num_observables,
                raw_num_paulis=[0] * num_observables,
                num_unique_paulis=[len(op) for op in observables_tmp]
                if operator_budget.simplify
                else None,
                num_duplicate_paulis=[0] * num_observables if operator_budget.simplify else None,
                num_trimmed_paulis=[0] * num_observables if operator_budget.simplify else None,
                sum_trimmed_coeffs=[0] * num_observables if operator_budget.simplify else None,
                num_truncated_paulis=[0] * num_observables,
                num_paulis=[0] * num_observables,
                sum_paulis=None,
                num_qwc_groups=None,
            )
            metadata.backpropagation_history.append(slice_metadata)

            # Iterate over all provided observables. We do this indirectly via their index because we
            # also need to access the respective entry in qargs and we want to overwrite the items of
            # the iterated lists in-place
            # PERF: we will likely need to parallelize this loop
            for i in range(num_observables):
                non_trivial_slice = False
                op_nodes = list(circuit_to_dag(slice_).topological_op_nodes())[::-1]
                for op_idx, op_node in enumerate(op_nodes):
                    # Ignore barriers within slices
                    if op_node.name == "barrier":
                        continue

                    op_qargs = [slice_.find_bit(q).index for q in op_node.qargs]

                    # NOTE: we do not need to take slice_qargs into account here because the op_nodes
                    # inside a slice are guaranteed to be disjoint. Thus, any op_nodes which have
                    # already been composed into slice_op cannot affect qargs_tmp_set in a way to cause
                    # it to overlap with the current op_qargs unless it already did before having
                    # applied any op_node from this slice.
                    if not set(qargs_tmp[i]).intersection(op_qargs):
                        # we can skip this gate if it does not intersect with the local observable
                        continue

                    non_trivial_slice = True

                    if op_node.name == "reset":
                        observables_tmp[i] = apply_reset_to(observables_tmp[i], qargs_tmp[i][0])
                    else:
                        # Absorb gate into observable and update qubits on which the observable acts
                        observables_tmp[i], qargs_tmp[i] = apply_op_to(
                            observables_tmp[i],
                            qargs_tmp[i],
                            SparsePauliOp.from_operator(Operator(op_node.op)),
                            op_qargs,
                            apply_as_transform=True,
                        )

                    slice_metadata.raw_num_paulis[i] = len(observables_tmp[i])

                    if operator_budget.simplify:
                        observables_tmp[i], simplify_metadata = simplify_sparse_pauli_op(
                            observables_tmp[i],
                            atol=operator_budget.atol,
                            rtol=operator_budget.rtol,
                        )
                        slice_metadata.num_unique_paulis[i] = (  # type: ignore[index]
                            simplify_metadata.num_unique_paulis
                        )
                        slice_metadata.num_duplicate_paulis[i] = (  # type: ignore[index]
                            simplify_metadata.num_duplicate_paulis
                        )
                        slice_metadata.num_trimmed_paulis[i] = (  # type: ignore[index]
                            simplify_metadata.num_trimmed_paulis
                        )
                        slice_metadata.sum_trimmed_coeffs[i] = (  # type: ignore[index]
                            simplify_metadata.sum_trimmed_coeffs
                        )

                    LOGGER.debug(
                        f"Size of the observable after backpropagating gate id {len(op_nodes) - op_idx - 1} in "
                        f"the current layer: {len(observables_tmp[i])}"
                    )

                if truncation_error_budget.is_active() and non_trivial_slice:
                    previous_size = len(observables_tmp[i])
                    observables_tmp[i] = _truncate_terms(
                        observables_tmp[i],
                        metadata,
                        i,
                    )
                    slice_metadata.num_truncated_paulis[i] = previous_size - len(observables_tmp[i])

                slice_metadata.num_paulis[i] = len(observables_tmp[i])

                LOGGER.info(
                    f"[{metadata.num_backpropagated_slices:3}] size of the {i}-th observable: "
                    f"{slice_metadata.num_paulis[i]}"
                )

            # gather up all unique Pauli terms across all observables
            # NOTE: we cannot rely on the length of an observable unless it was simplified. Moreover, we
            # also cannot rely on the sum of all observables since coefficients might turn to become
            # zero. Plus, since we do not care about the coefficients here, we can avoid doing the math
            # and simply gather the set of all unique Pauli terms.
            num_qubits = slices[0].num_qubits
            if operator_budget.is_active():  # noqa: SIM102
                if _observable_oversized(
                    observables_tmp,
                    qargs_tmp,
                    num_qubits,
                    operator_budget,
                    metadata.num_backpropagated_slices,
                    slice_metadata,
                ):
                    break

            # If we have reached this point, we have successfully backpropagated a slice within our
            # thresholds and reflect this by updating our output variables
            observables_out = copy.deepcopy(observables_tmp)
            qargs_out = copy.deepcopy(qargs_tmp)

            # Finally, we increment the backpropagated slice count. This is done regardless of whether
            # this slice contained gates which were overlapping with our observables or not. However, it
            # is done only for slices which did actually contain a gate of any kind.
            metadata.num_backpropagated_slices += 1

    # Catch the TimeoutException which the max_seconds alarm may trigger
    except TimeoutException:  # pragma: no cover
        LOGGER.warning("Reached the specified max_seconds timeout!")  # pragma: no cover
        pass  # pragma: no cover

    finally:
        # Override signal alarm to deactivate it.
        if max_seconds is not None:
            signal.alarm(0)

    LOGGER.info(f"Backpropagated {metadata.num_backpropagated_slices} DAG multi-graph slices")

    # Return the slices which were not backpropagated
    slices_out = slices[: -metadata.num_backpropagated_slices]

    # Specify the output observables on all qubits in the circuit
    obs_out = [
        to_global_op(obs, qargs, num_qubits) for obs, qargs in zip(observables_out, qargs_out)
    ]
    # If the input was a single observable, return a single observable
    if isinstance(observables, SparsePauliOp):
        assert len(obs_out) == 1
        obs_out = obs_out[0]

    return obs_out, slices_out, metadata


def _validate_input_options(
    global_observables: SparsePauliOp | Sequence[SparsePauliOp],
    slices: Sequence[QuantumCircuit],
    operator_budget: OperatorBudget,
):
    """Ensure input arguments to OBP are valid."""
    if isinstance(global_observables, SparsePauliOp):
        global_observables = [global_observables]
    num_qubits = slices[0].num_qubits
    for i, slice_ in enumerate(slices):
        if slice_.num_qubits != num_qubits:
            raise ValueError(
                "All slices must be defined on the same number of qubits. "
                f"slices[0] contains {num_qubits} qubits, but slices[{i}] contains "
                f"{slice_.num_qubits} qubits."
            )
    obs_qubits = global_observables[0].num_qubits
    for i, obs in enumerate(global_observables):
        if obs.num_qubits != obs_qubits:
            raise ValueError(
                "Input observables must all act on the same number of qubits. "
                f"observables[0] acts on {obs_qubits} qubits, but observables[{i}]"
                f" acts on {obs.num_qubits} qubits."
            )
    if obs_qubits != num_qubits:
        raise ValueError(
            "The input observables must be defined on the same number of qubits as the "
            "circuit slices."
        )
    max_paulis = operator_budget.max_paulis
    max_qwc_groups = operator_budget.max_qwc_groups
    if max_paulis is not None and max_paulis < 1:
        raise ValueError("Limiting the number of Pauli terms to less than 1 does not make sense.")
    if max_qwc_groups is not None and max_qwc_groups < 1:
        raise ValueError(
            "Limiting the number of qubit-wise commmuting Pauli groups to less than 1 does not "
            "make sense."
        )
    if operator_budget.is_active():
        # combine all observables as global ones into one large one
        all_paulis = SparsePauliOp.from_list([], num_qubits=num_qubits)
        for obs in global_observables:
            all_paulis += SparsePauliOp.from_list([(pauli, 1) for pauli, _ in obs.label_iter()])
        all_paulis = all_paulis.simplify()

        if max_paulis is not None:
            num_paulis = len(all_paulis)
            if num_paulis > max_paulis:
                raise ValueError(
                    f"You specified a maximum number of Pauli terms of {max_paulis}, but the "
                    f"provided observables already exceed this threshold with a total of "
                    f"{num_paulis} terms."
                )

        if max_qwc_groups is not None:
            num_qwc_groups = len(all_paulis.group_commuting(qubit_wise=True))
            if num_qwc_groups > max_qwc_groups:
                raise ValueError(
                    f"You specified a maximum number of qubit-wise commuting Pauli groups of "
                    f"{max_qwc_groups}, but the provided observables already exceed this threshold "
                    f"with a total of {num_qwc_groups} terms."
                )


def _truncate_terms(
    observable: SparsePauliOp,
    metadata: OBPMetadata,
    observable_idx: int,
) -> SparsePauliOp:
    """Truncate terms from the observable and update the budgets."""
    slice_idx = metadata.num_backpropagated_slices
    p_norm = metadata.truncation_error_budget.p_norm

    accumulated_error = metadata.accumulated_error(observable_idx, slice_idx)
    left_over_error_budget = metadata.left_over_error_budget(observable_idx, slice_idx)

    # Truncate low-weight observable terms
    observable_out, slice_error = truncate_binary_search(
        observable,
        left_over_error_budget,
        p_norm=p_norm,
        tol=metadata.truncation_error_budget.tol,
    )

    accumulated_error = metadata.accumulated_error(observable_idx, slice_idx + 1)
    LOGGER.info(
        f"[{slice_idx:3}] Accumulated error for the {observable_idx}-th observable: "
        f"{accumulated_error:.10f}"
    )

    metadata.backpropagation_history[slice_idx].slice_errors[observable_idx] = slice_error

    return observable_out


def _observable_oversized(
    observables: Sequence[SparsePauliOp],
    qargs: list[list[int]],
    num_qubits: int,
    operator_budget: OperatorBudget,
    slice_id: int,
    slice_metadata: SliceMetadata,
) -> bool:
    """Check if observable has overgrown the specified bounds."""
    # combine all observables as global ones into one large one
    all_paulis = SparsePauliOp.from_list([], num_qubits=num_qubits)
    max_paulis = operator_budget.max_paulis
    max_qwc_groups = operator_budget.max_qwc_groups
    for obs, qargs_ in zip(observables, qargs):
        global_obs = to_global_op(obs, qargs_, num_qubits)
        all_paulis += SparsePauliOp.from_list([(pauli, 1) for pauli, _ in global_obs.label_iter()])
    all_paulis = all_paulis.simplify()

    if max_paulis is not None:
        slice_metadata.sum_paulis = len(all_paulis)
        if slice_metadata.sum_paulis > max_paulis:
            LOGGER.info(f"[{slice_id:3}] Too many Pauli terms: {slice_metadata.sum_paulis}")
            return True

    if max_qwc_groups is not None:
        slice_metadata.num_qwc_groups = len(all_paulis.group_commuting(qubit_wise=True))
        if slice_metadata.num_qwc_groups > max_qwc_groups:
            LOGGER.info(
                f"[{slice_id:3}] Too many qubit-wise commuting Pauli groups: "
                f"{slice_metadata.num_qwc_groups}"
            )
            return True

    return False


def _get_observable_and_qargs_lists(
    observables: SparsePauliOp | Sequence[SparsePauliOp],
) -> tuple[list[SparsePauliOp], list[list[int]]]:
    """Ensure observables and qargs are lists."""
    observable_list = [observables] if isinstance(observables, SparsePauliOp) else list(observables)

    # Get lean representations of the observables (reduced obs + qargs)
    obs_out = []
    qargs_out = []
    for observable in observable_list:
        reduced_op = reduce_op(observable)
        obs_out.append(reduced_op[0])
        qargs_out.append(reduced_op[1])

    return obs_out, qargs_out
