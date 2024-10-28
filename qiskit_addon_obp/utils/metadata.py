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
"""Container classes for holding backpropagation metadata."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from .simplify import OperatorBudget
from .truncating import TruncationErrorBudget


@dataclass
class SliceMetadata:
    """A container for metadata generated during the backpropagation of a single slice."""

    slice_errors: list[float]
    """The truncation error incurred for each observable that is being backpropagated."""

    raw_num_paulis: list[int]
    """The "raw" number of Pauli terms for each observable that is being backpropagated. This
    indicates the number of Pauli terms before any simplification or truncation operations."""

    num_unique_paulis: list[int] | None
    """The number of unique Pauli terms for each observable. This attribute only tracks those Pauli
    terms whose coefficients were not already below the trimming threshold at the beginning of the
    operator simplification procedure.

    .. note::
       This value will be ``None`` if ``simplify=False`` during the backpropagation.
    """

    num_duplicate_paulis: list[int] | None
    """The number of duplicate Pauli terms for each observable whose coefficients have been summed
    up to trim excess memory usage during the operator simplification.

    .. note::
       This value will be ``None`` if ``simplify=False`` during the backpropagation.
    """

    num_trimmed_paulis: list[int] | None
    """The number of Pauli terms for each observable that were trimmed during operator
    simplification because their total coefficient was below the trimming threshold.

    .. note::
       This value will be ``None`` if ``simplify=False`` during the backpropagation.
    """

    sum_trimmed_coeffs: list[float] | None
    """The sum of the coefficients for each observable that were trimmed during operator
    simplification because each individual coefficient was below the trimming threshold.

    .. warning::
       This sum is *not* affected by the value of :attr:`.p_norm`!

    .. note::
       This value will be ``None`` if ``simplify=False`` during the backpropagation.
    """

    num_truncated_paulis: list[int]
    """The number of Pauli terms which were truncated from each observable that is being
    backpropagated."""

    num_paulis: list[int]
    """The number of Pauli terms for each observable that is being backpropagated."""

    sum_paulis: int | None
    """The number of Pauli terms across all observables. This value is ``None`` if it is not bounded
    during the backpropagation.

    .. note::
       This value can differ from ``sum(num_paulis)`` because identical Pauli terms will not be
       counted multiple times.
    """

    num_qwc_groups: int | None
    """The number of qubit-wise commuting groups of Pauli terms across all observables. This value
    is ``None`` if it is not bounded during the backpropagation."""


@dataclass
class OBPMetadata:
    """A container for metadata generated during the :meth:`backpropagate` method."""

    truncation_error_budget: TruncationErrorBudget
    """Values specifying the observable truncation strategy."""

    num_slices: int | None
    """The total number of slices to attempt to backpropagate."""

    operator_budget: OperatorBudget
    """Values specifying how large the operator may grow."""

    backpropagation_history: list[SliceMetadata]
    """A sequence of metadata objects to track the progression of the backpropagation."""

    num_backpropagated_slices: int
    """The number of backpropagated slices.

    .. note::
       This value can differ from ``len(backpropagation_history)`` in cases where a stopping
       criterion such as ``operator_budget.max_paulis`` or ``operator_budget.max_qwc_groups``
       caused the backpropagation to terminate. Then, this value should be equal to
       ``len(backpropagation_history) - 1`` because the last slice caused the thresholds to
       be exceeded. However, for convenience these values are still recorded for the end-user
       to inspect them.
    """

    def accumulated_error(self, observable_idx: int, slice_idx: int | None = None) -> float:
        """Compute the accumulated error for a given observable at a given "time".

        This method computes the accumulated error for a given observable index at a given "time"
        during the course of the backpropagation. In this context, "time" is to be understood as
        the discrete steps of already backpropagated slices.

        The accumulated error is computed as the sum of the individual
        :attr:`.SliceMetadata.slice_errors`. These in turn may be computed within a specified
        :attr:`.TruncationErrorBudget.p_norm`. Thus, the computed accumulated error is an upper bound
        to the real accumulated error as given by the `Minkowski inequality
        <https://en.wikipedia.org/wiki/Minkowski_inequality>`_ (the generalization of the triangle
        inequality for Lp-norms other than ``p=2``).

        .. note::
           Since a general Lp-norm (other than ``p=2``) is *not* an inner product norm, it does
           *not* satisfy the `parallelogram law <https://en.wikipedia.org/wiki/Parallelogram_law>`_.
           Hence, we must use the Minkowski inequality as the upper bound of the accumulated error.

        Args:
            observable_idx: the index of the observable whose accumulated error to compute.
            slice_idx: the index of the slice (the discrete "time" step) up to which to compute the
                accumulated error. If this is ``None``, it will default to
                ``self.num_backpropagated_slices`` which is equivalent to computing the accumulated
                error of an observable at the latest point in time which has not surpassed any
                stopping criteria in the algorithm

        Returns:
            The accumulated error computed per the explanations above.

        """
        if slice_idx is None:
            slice_idx = self.num_backpropagated_slices  # pragma: no cover

        accumulated_error = 0.0
        for i in range(slice_idx):
            accumulated_error += self.backpropagation_history[i].slice_errors[observable_idx]

        return accumulated_error

    def left_over_error_budget(self, observable_idx: int, slice_idx: int | None = None) -> float:
        """Compute the left-over error budget for a given observable at a given "time".

        This method computes the left-over error budget for a given observable index at a given
        "time" during the course of the backpropagation. In this context, "time" is to be
        understood as the discrete steps of already backpropagated slices.

        The left-over error budget is computed as the remainder of the total budget minus the sum
        of the individual :attr:`.SliceMetadata.slice_errors`. These in turn may be computed
        within a specified :attr:`.TruncationErrorBudget.p_norm`.

        .. note::
           See also the explanations in :meth:`accumulated_error` for more details on how the
           individual slice errors are summed up to form an upper bound to the real error via the
           Minkowski inequality.

        .. note::
            The left-over error budget is relative to the value of
            ``self.truncation_error_budget.max_error_total`` which may be ``numpy.inf`` in which
            case the returned value of this method will be ``numpy.inf``, too.

        Args:
            observable_idx: the index of the observable whose left-over error budget to compute.
            slice_idx: the index of the slice (the discrete "time" step) up to which to compute the
                left-over error budget. If this is ``None``, it will default to
                ``self.num_backpropagated_slices`` which is equivalent to computing the left-over
                error budget of an observable at the latest point in time which has not surpassed
                any stopping criteria in the algorithm

        Returns:
            The left-over error budget computed per the explanations above.

        """
        if slice_idx is None:
            slice_idx = self.num_backpropagated_slices  # pragma: no cover

        left_over_error_budget = 0.0
        for i in range(slice_idx + 1):
            # NOTE: we are iterating up-to `slice_idx + 1` here because we want to include the
            # budget that became available at the backpropagation of this current slice (i.e. we
            # want `range` to be _inclusive_ of the provided `slice_idx`)
            slice_budget = self.truncation_error_budget.per_slice_budget[
                i % len(self.truncation_error_budget.per_slice_budget)
            ]

            # increment the budget for this iteration
            left_over_error_budget = left_over_error_budget + slice_budget

            # remove any budget which has already been used up
            if i > 0:
                left_over_error_budget -= self.backpropagation_history[i - 1].slice_errors[
                    observable_idx
                ]

            # prevent the budget from exceeding the true total budget
            left_over_error_budget = min(
                left_over_error_budget,
                self.truncation_error_budget.max_error_total
                - self.accumulated_error(observable_idx, i),
            )

        return left_over_error_budget

    @classmethod
    def from_json(cls, json_file: str) -> OBPMetadata:
        """Load a metadata from a json file.

        Args:
            json_file: the path to the file to be loaded.

        Returns:
            The loaded metadata.

        """
        with open(json_file) as file:
            data = json.load(file)

        metadata = cls(**data)
        metadata.truncation_error_budget = TruncationErrorBudget(  # type: ignore[arg-type]
            **metadata.truncation_error_budget
        )
        metadata.operator_budget = OperatorBudget(**metadata.operator_budget)  # type: ignore[arg-type]
        metadata.backpropagation_history = [
            SliceMetadata(**raw_slice_data)  # type: ignore[arg-type]
            for raw_slice_data in metadata.backpropagation_history
        ]

        return metadata

    def to_json(self, json_file: str, **kwargs) -> None:
        """Dump this metadata to a json file.

        Args:
            json_file: the path to the file into which to dump the metadata.
            kwargs: keyword arguments to be passed on towards :meth:`.json.dump`.

        """
        with open(json_file, "w") as file:
            json.dump(asdict(self), file, **kwargs)
