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

# Reminder: update the RST file in docs/apidocs when adding new interfaces.
"""Utility functions for operator backpropagation."""

from __future__ import annotations

from qiskit.quantum_info import SparsePauliOp


def apply_op_to(
    op1: SparsePauliOp,
    op1_qargs: list[int],
    op2: SparsePauliOp,
    op2_qargs: list[int],
    *,
    apply_as_transform: bool = False,
) -> tuple[SparsePauliOp, list[int]]:
    """Apply the operator ``op2`` to the operator ``op1``.

    These operators do not necessarily need to act on the same number of qubits, as they
    are assumed to act on a larger system. The position in the system of each operator
    is defined by the corresponding ``qargs``. The output operator will be defined
    on ``union(op1_qargs, op2_qargs)``.

    By default, this function applies ``op1`` to ``op2`` in the following way:

        ``op2 @ op1``

    By setting ``apply_as_transform=True``, this function will apply ``op1`` to ``op2``
    in the following way:

        ``op2.adjoint() @ op1 @ op2``

    Args:
        op1: The operator on which ``op2`` will be applied.
        op1_qargs: The qubit indices for ``op1``.
        op2: The operator to apply to ``op1``.
        op2_qargs: The qubit indices for ``op2``.
        apply_as_transform: Whether to apply ``op2`` to ``op1`` as a transformation.

    Returns:
        The tuple ``(op, qargs)`` where ``op`` is the input ``op1`` with ``op2`` left-applied
        and ``qargs`` is a list of qubit indices for the new operator ``op``. The qubit IDs
        in the output ``op`` correspond to the global qubit ID in the same index in ``qargs``.

        For example, if the output ``op`` is a ``SparsePauliOp("YX")`` and ``qargs`` is [3, 4],
        the X term on qubit 0 of the operator corresponds to global qubit ID 3.

    Raises:
        ValueError: The number of unique operator qargs must match the number of qubits in the
            corresponding operator.

    """
    _validate_qargs(op1, op1_qargs)
    _validate_qargs(op2, op2_qargs)

    if set(op2_qargs) <= set(op1_qargs):
        # all of the qargs of op2 are already covered by op1
        op1_expanded = op1
        op1_qargs_out = op1_qargs
    else:
        op1_qargs_out = list(sorted(set(op1_qargs).union(set(op2_qargs))))
        num_qubits = len(op1_qargs_out)

        # PERF: if the `.compose` call inside of `.apply_layout` becomes a bottleneck, it might be
        # possible to improve performance for the cases in which we only need to pre-/append qubits
        # on either end of the operators (i.e. rather than insert them in the middle).
        op1_expanded = op1.apply_layout([op1_qargs_out.index(q) for q in op1_qargs], num_qubits)

    op2_qargs_in_op1 = [op1_qargs_out.index(q) for q in op2_qargs]

    if apply_as_transform:
        op_out = op1_expanded.compose(op2, qargs=op2_qargs_in_op1, front=True).compose(
            op2.adjoint(), qargs=op2_qargs_in_op1
        )
    else:
        op_out = op1_expanded.compose(op2, qargs=op2_qargs_in_op1)

    return op_out, op1_qargs_out


def to_global_op(op: SparsePauliOp, qargs: list[int], n_qubits: int) -> SparsePauliOp:
    """Convert a local operator to a global operator by inserting identities on qubits which aren't used.

    Args:
        op: Local operator to expand.
        qargs: Qubit indices for local operator.
        n_qubits: Number of qubits in the global system.

    Returns:
        An operator on ``n_qubits`` qubits

    Raises:
        ValueError: Qubit ID out of range

    """
    min_qargs = min(qargs)
    max_qargs = max(qargs)
    if min_qargs < 0:
        raise ValueError(f"qargs may not contain a negative qubit ID. Found: ({min_qargs}).")
    if n_qubits <= max_qargs:
        raise ValueError(
            f"qargs contains qubit ID ({max_qargs}), but the global system contains only ({n_qubits}) qubits."
        )
    new_qargs = list(range(n_qubits))
    return op.apply_layout([new_qargs.index(q) for q in qargs], n_qubits)


def _validate_qargs(op: SparsePauliOp, qargs: list[int]) -> None:
    if op.num_qubits != len(set(qargs)):
        raise ValueError(
            f"The number of qubits in the operator ({op.num_qubits}) does not match "
            f"the number of qargs ({len(set(qargs))})."
        )


def reduce_op(global_op: SparsePauliOp) -> tuple[SparsePauliOp, list[int]]:
    """Create a lean representation of a global Pauli operator.

    This function returns a lean representation of the input operator such that all
    of the qubits associated solely with Pauli-I terms have been removed. A list
    of indices is also returned indicating on which qubits the lean operator acts.

    For example:
        >>> global_op = SparsePauliOp(["IXI", "IIZ"])
        >>> reduced_op, qargs = reduce_op(global_op)
        >>> reduced_op
        SparsePauliOp(['XI', 'IZ'],
                      coeffs=[1.+0.j, 1.+0.j])
        >>> qargs
        [0, 1]

    Args:
        global_op: The global operator for which to generate a lean representation

    Returns:
        - A lean representation of the input operator with qubits associated solely
          with identity terms removed.
        - A list of indices specifying the qubits on which the lean operator acts.

    Raises:
        ValueError: Input operator may not be the identity operator.

    """
    pauli_strings = [p.to_label() for p in global_op.paulis]
    reduced_qargs = [
        int(q) for q in (global_op.paulis.x + global_op.paulis.z).sum(axis=0).nonzero()[0]
    ]

    if reduced_qargs == []:
        raise ValueError("Input operator may not be the identity operator.")

    reduced_op = SparsePauliOp(
        [
            "".join(s[global_op.num_qubits - i - 1] for i in reduced_qargs[::-1])
            for s in pauli_strings
        ],
        coeffs=global_op.coeffs,
    )

    return reduced_op, reduced_qargs


def apply_reset_to(
    op: SparsePauliOp,
    qubit_id: int,
    inplace: bool = False,
) -> SparsePauliOp:
    """Apply a reset operation to a Pauli operator.

    This function applies a reset operation to ``op`` in the following way:

        ``<0|op|0>``

    Terms containing Pauli X or Y terms on qubit-``qubit_id`` will have their
    weight reduced to ``0.0``. Terms containing Pauli Z on ``qubit_id`` will
    have that Pauli Z replaced by an identity.

    Args:
        op: The operator to which the reset will be applied.
        qubit_id: The index of the qubit on which to apply the reset.
        inplace: Whether to modify the operator in-place.

    Returns:
        The transformed operator

    """
    if not inplace:
        op = op.copy()

    op.coeffs[op.paulis.x[:, qubit_id]] = 0.0
    op.paulis.z[:, qubit_id] = False
    op.paulis.x[:, qubit_id] = False

    return op
