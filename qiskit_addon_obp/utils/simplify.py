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
"""Functions for simplifying Pauli operators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit._accelerate.sparse_pauli_op import unordered_unique
from qiskit.circuit import ParameterExpression
from qiskit.quantum_info import PauliList, SparsePauliOp


@dataclass
class SimplifyMetadata:
    """A simple dataclass for returning the tracked attributes during operator simplification."""

    num_unique_paulis: int
    """The number of unique Pauli terms. See also :attr:`~SliceMetadata.num_unique_paulis`."""

    num_duplicate_paulis: int
    """The number of duplicate Pauli terms. See also :attr:`~SliceMetadata.num_duplicate_paulis`."""

    num_trimmed_paulis: int
    """The number of trimmed Pauli terms. See also :attr:`~SliceMetadata.num_trimmed_paulis`."""

    sum_trimmed_coeffs: float
    """The sum of the trimmed coefficients. See also :attr:`~SliceMetadata.sum_trimmed_coeffs`."""


@dataclass
class OperatorBudget:
    """A class for storing the constants that determine how large an operator may grow.

    Backpropagation will stop if either of the following conditions is met:

    - The number of Pauli terms across all of the observables exceeds ``max_paulis``. When
      ``max_paulis = None``, the number of Pauli terms in the observables is not constrained.
    - The number of qubit-wise commuting Pauli groups across all of the observables exceed
      ``max_qwc_groups``. When ``max_qwc_groups = None``, the number of qubit-wise commuting Pauli
      groups in the observables is not constrained.
    """

    max_paulis: int | None = None
    """The maximum number of Pauli terms the backpropagated operator may contain."""

    max_qwc_groups: int | None = None
    """The maximum number of qubit-wise commuting Pauli groups the backpropagated
    operator may contain.
    """

    simplify: bool = True
    """A flag denoting whether to call :func:`simplify` at every iteration."""

    atol: float | None = None
    """Absolute tolerance for checking if coefficients are zero. Defaults to the
    :attr:`~qiskit.quantum_info.SparsePauliOp.atol` value of ``SparsePauliOp``."""

    rtol: float | None = None
    """Relative tolerance for checking if coefficients are zero. Defaults to the
    :attr:`~qiskit.quantum_info.SparsePauliOp.rtol` value of ``Sparse``."""

    def is_active(self) -> bool:
        """Return whether ``self`` places any bounds on operator size."""
        return self.max_paulis is not None or self.max_qwc_groups is not None


def simplify(
    operator: SparsePauliOp,
    *,
    atol: float | None = None,
    rtol: float | None = None,
) -> tuple[SparsePauliOp, SimplifyMetadata]:
    """Simplifies the provided Pauli operator.

    This is an adaption of :meth:`.SparsePauliOp.simplify` which tracks metadata of the simplified
    terms.

    Args:
        operator: the :class:`.SparsePauliOp` to simplify.
        atol: the absolute tolerance for checking if coefficients are zero. If ``None``, this will
            fallback to using :attr:`.SparsePauliOp.atol`.
        rtol: the relative tolerance for checking if coefficients are zero. If ``None``, this will
            fallback to using :attr:`.SparsePauliOp.rtol`.

    Returns:
        The simplified Pauli operator.

    """
    # Get default atol and rtol
    if atol is None:
        atol = operator.atol
    if rtol is None:
        rtol = operator.rtol

    def to_complex(coeff):
        if not hasattr(coeff, "sympify"):
            return coeff
        sympified = coeff.sympify()
        return complex(sympified) if sympified.is_Number else np.nan

    # Filter non-zero coefficients
    if operator.coeffs.dtype == object:
        non_zero = np.logical_not(
            np.isclose([to_complex(x) for x in operator.coeffs], 0, atol=atol, rtol=rtol)
        )
    else:
        non_zero = np.logical_not(np.isclose(operator.coeffs, 0, atol=atol, rtol=rtol))
    paulis_x = operator.paulis.x[non_zero]
    paulis_z = operator.paulis.z[non_zero]
    nz_coeffs = operator.coeffs[non_zero]

    array = np.packbits(paulis_x, axis=1).astype(np.uint16) * 256 + np.packbits(paulis_z, axis=1)
    indexes, inverses = unordered_unique(array)

    metadata = SimplifyMetadata(
        num_unique_paulis=len(indexes),
        num_duplicate_paulis=len(nz_coeffs) - len(indexes),
        num_trimmed_paulis=len(operator.coeffs) - len(nz_coeffs),
        sum_trimmed_coeffs=0.0,
    )

    if np.all(non_zero) and indexes.shape[0] == array.shape[0]:
        # No zero operator or duplicate operator
        return operator.copy(), metadata

    coeffs = np.zeros(indexes.shape[0], dtype=operator.coeffs.dtype)
    np.add.at(coeffs, inverses, nz_coeffs)
    # Delete zero coefficient rows
    if operator.coeffs.dtype == object:
        is_zero = np.array(
            [np.isclose(to_complex(coeff), 0, atol=atol, rtol=rtol) for coeff in coeffs]
        )
    else:
        is_zero = np.isclose(coeffs, 0, atol=atol, rtol=rtol)

    metadata.num_trimmed_paulis += sum(is_zero)
    metadata.sum_trimmed_coeffs += sum(coeffs[is_zero])
    if isinstance(metadata.sum_trimmed_coeffs, ParameterExpression):
        metadata.sum_trimmed_coeffs = to_complex(metadata.sum_trimmed_coeffs)

    # Check edge case that we deleted all Paulis
    # In this case we return an identity Pauli with a zero coefficient
    if np.all(is_zero):
        x = np.zeros((1, operator.num_qubits), dtype=bool)
        z = np.zeros((1, operator.num_qubits), dtype=bool)
        coeffs = np.array([0j], dtype=operator.coeffs.dtype)
    else:
        non_zero = np.logical_not(is_zero)
        non_zero_indexes = indexes[non_zero]
        x = paulis_x[non_zero_indexes]
        z = paulis_z[non_zero_indexes]
        coeffs = coeffs[non_zero]

    return SparsePauliOp(
        PauliList.from_symplectic(z, x), coeffs, ignore_pauli_phase=True, copy=False
    ), metadata
