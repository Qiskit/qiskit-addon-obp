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
"""Functions for truncating Pauli operators within given error budgets."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from qiskit.quantum_info import SparsePauliOp


@dataclass
class TruncationErrorBudget:
    """A class for storing the constants that determine the truncation error budget.

    Refer to the `how-to guide <https://qiskit.github.io/qiskit-addon-obp/how_tos/truncate_operator_terms.html>`__ for a detailed discussion
    on truncating operator terms during backpropagation and bounding the incurred error.
    """

    per_slice_budget: list[float] = field(default_factory=list)
    """The maximum amount of truncation error to allow per backpropagated slice. This list will be
    looped over during the backpropagation of the circuit slices.
    """

    max_error_total: float = 0.0
    """The maximum total truncation error to allow for each observable during the entire
    backpropagation. This value may be :external:obj:`numpy.inf`, for example when the maximum
    error was left unspecified during :meth:`.setup_budget`."""

    p_norm: int = 1
    """
    Indicates which Lp-norm is used for calculating truncation errors.

    Refer to the `how-to guide <https://qiskit.github.io/qiskit-addon-obp/how_tos/bound_error_using_p_norm.html>`__ for a detailed
    conversation on bounding truncation error using higher Lp-norms.
    """

    tol: float = 1e-8
    """Absolute tolerance used during truncation. Once an optimal truncation threshold, up
    to this tolerance, has been found, the search for an optimal threshold will stop.
    """

    def is_active(self) -> bool:
        """Return whether the truncation is active, i.e. whether the budget is non-zero."""
        return any(budget > 0 for budget in self.per_slice_budget) and self.max_error_total > 0


def setup_budget(
    *,
    max_error_per_slice: float | Sequence[float] | None = None,
    max_error_total: float | None = None,
    num_slices: int | None = None,
    p_norm: int = 1,
) -> TruncationErrorBudget:
    """Calculate the budget available to each slice for observable term truncation.

    This method makes the construction of a :class:`.TruncationErrorBudget` easier for an end-user.
    This error budget can be provided to the :meth:`~qiskit_addon_obp.backpropagate` method to
    enable the truncation of low-weight Pauli terms. Refer to the `how-to guide
    <https://qiskit.github.io/qiskit-addon-obp/how_tos/truncate_operator_terms.html>`__ for a detailed discussion on truncating terms from
    the output operator and bounding the incurred error.

    The construction logic is as follows:
        - if ``max_error_per_slice`` is provided its value is converted to a list and used immediately
          for :attr:`.TruncationErrorBudget.per_slice_budget`
        - if the above is not the case, ``max_error_total`` has to be set
        - if ``num_slices`` is **not** set,:attr:`.TruncationErrorBudget.per_slice_budget` gets set to
          ``[max_error_total]`` resulting in the entire budget being consumed **greedily**
        - however, if ``num_slices`` is provided, then :attr:`.TruncationErrorBudget.per_slice_budget`
          assumes an even distribution of the maximum total error across the specified number of
          slices: ``[max_error_total / num_slices]``

    Finally, if ``max_error_total`` is set, this puts a hard limit on the total maximum error to be
    accumulated throughout the entire backpropagation. Thus, setting ``max_error_per_slice`` and
    ``max_error_total`` can be of value.

    .. note::

       Budget not spent during a previous iteration will carry over into subsequent iterations,
       meaning that the maximum budget for any slice during backpropagation may in fact exceed
       :attr:`.TruncationErrorBudget.per_slice_budget`.

    Args:
        max_error_per_slice: Specifies the maximum error per backpropagated slice. See above for
            more details.
        max_error_total: Specifies the total maximum error for the entire backpropagation. See
            above for more details.
        num_slices: The number of slices over which to distribute the budget. See above for more details.
        p_norm: The Lp norm of the error. This affects the gradual distribution of
            ``max_error_total`` in the case of ``num_slices`` also being set (see above). Refer to the
            `how-to guide <https://qiskit.github.io/qiskit-addon-obp/how_tos/bound_error_using_p_norm.html>`__ for a detailed conversation
            on bounding truncation error using higher Lp-norms.

    Returns:
        The resulting :class:`.TruncationErrorBudget`.

    Raises:
        ValueError: if ``max_error_per_slice`` and ``max_error_total`` are both ``None``.

    """
    if max_error_per_slice is None and max_error_total is None:
        raise ValueError("max_error_per_slice and max_error_total may not both be None")

    per_slice_budget: list[float] = []

    if max_error_per_slice is not None:
        # the simplest case: the user provided the direct configuration input for this setting
        if isinstance(max_error_per_slice, float):
            per_slice_budget = [max_error_per_slice]
        else:
            per_slice_budget = list(max_error_per_slice)

    elif max_error_total is not None:
        # otherwise, if a maximum total error was set, we have two cases:
        if num_slices is not None:
            # we divide the budget evenly across the number of slices
            per_slice_budget = [max_error_total / num_slices]
        else:
            # we greedily use up the total error budget at each slice
            per_slice_budget = [max_error_total]
    else:  # pragma: no cover
        # NOTE: this code is unreachable because it matches the `ValueError` being raised above
        pass

    if max_error_total is None:
        max_error_total = np.inf

    return TruncationErrorBudget(per_slice_budget, max_error_total, p_norm)


def truncate_binary_search(
    observable: SparsePauliOp,
    budget: float,
    *,
    p_norm: int = 1,
    tol: float = 1e-8,
) -> tuple[SparsePauliOp, float]:
    r"""Perform binary search to find an optimal observable truncation threshold.

    Removes the Pauli terms from a ``SparsePauliOp`` whose sum of their absolute coefficients values
    does not exceed the provided error ``budget``.

    Args:
        observable: the ``SparsePauliOp`` to truncate terms from.
        budget: the maximum permissable truncation error.
        p_norm: an integer specifying what p-norm to use.
        tol: when the binary search thresholds differ by an amount smaller than ``tol``, the
            threshold search will stop.

    Returns:
        The truncated observable and a bound on the incurred truncation error.

        .. note::
           The incurred truncation error bound, :math:`E`, is calculated as the ``p-norm`` of the
           truncated terms' coefficient magnitudes, :math:`c`, such that :math:`E = \|c\|_p`.

    """
    abscs = np.abs(observable.coeffs) ** p_norm

    upper_threshold = max(abscs)
    lower_threshold = 0.0
    upper_error = budget
    lower_error = 0.0

    # binary search for a cutoff threshold
    while ((upper_threshold - lower_threshold) > tol) and not (
        np.isclose(upper_error, lower_error, atol=tol)
    ):
        mid_threshold = (upper_threshold + lower_threshold) / 2
        # PERF: the boolean indexing here will need to check every element in the array at every
        # iteration of this loop. We can improve the performance by performing successive
        # partitioning of the array (see `numpy.partition` and `numpy.argpartition`) and only
        # considering the currently relevant sector of the binary search.
        mid_error = np.power(np.sum(abscs[abscs < mid_threshold]), 1.0 / p_norm)

        if mid_error <= budget:
            # mid_threshold is not high enough, so we should update lower_threshold to this value
            lower_threshold = mid_threshold
            lower_error = mid_error
        else:
            # mid_threshold is too high, update the upper_threshold to this value
            upper_threshold = mid_threshold
            upper_error = mid_error

    # delete according to lower_threshold
    to_keep = abscs > lower_threshold

    return (
        SparsePauliOp(observable.paulis[to_keep], observable.coeffs[to_keep]),
        lower_error,
    )
