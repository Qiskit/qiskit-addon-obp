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
"""Various visualization utilities."""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes

from .metadata import OBPMetadata


def plot_accumulated_error(metadata: OBPMetadata, axes: Axes, *, show_legend: bool = True) -> None:
    """Plot the accumulated error.

    This method populates the provided figure axes with a line-plot of the
    :meth:`.OBPMetadata.accumulated_error`. Below is an example where we plot some ``metadata``
    which exists within our context.

    .. plot::
       :alt: Output from the previous code.
       :context:
       :nofigs:

       >>> from qiskit_addon_obp.utils.metadata import OBPMetadata
       >>> metadata = OBPMetadata.from_json("docs/_static/dummy_visualization_metadata.json")

    .. plot::
       :alt: Output from the previous code.
       :context: close-figs
       :include-source:

       >>> from matplotlib import pyplot as plt
       >>> from qiskit_addon_obp.utils.visualization import plot_accumulated_error
       >>> fig, axes = plt.subplots(1, 1)
       >>> plot_accumulated_error(metadata, axes)

    As you can see in the figure above, the number of backpropagated slices is displayed along the
    x-axis. You can think of this as the "time" of the backpropagation algorithm. The accumulated
    error due to truncated Pauli terms is displayed along the y-axis. If
    :attr:`.OBPMetadata.truncation_error_budget.max_error_total` is not :external:obj:`numpy.inf`,
    it is displayed as a red horizontal line.
    Each observable that was backpropagated, gets its own line.

    This data is related to the one visualized by :meth:`plot_slice_errors`. This method plots the
    cumulative sum of the slice errors along the x-axis.

    Args:
        metadata: the metadata to be visualized.
        axes: the matplotlib axes in which to plot.
        show_legend: enable/disable showing the legend in the plot.
    """
    if not np.isinf(metadata.truncation_error_budget.max_error_total):
        axes.axhline(
            metadata.truncation_error_budget.max_error_total,
            label="max_error_total",
            color="red",
            alpha=0.75,
        )
    else:  # pragma: no cover
        pass
    for obs_idx in range(len(metadata.backpropagation_history[0].slice_errors)):
        axes.plot(
            range(1, len(metadata.backpropagation_history) + 1),
            [
                metadata.accumulated_error(obs_idx, i)
                for i in range(1, len(metadata.backpropagation_history) + 1)
            ],
            label=f"Observable #{obs_idx + 1}",
        )
    axes.set_xlabel("backpropagated slice number")
    axes.set_ylabel("accumulated error")
    _set_legend(axes, show_legend)


def plot_left_over_error_budget(
    metadata: OBPMetadata, axes: Axes, *, show_legend: bool = True
) -> None:
    """Plot the left-over error budget.

    This method populates the provided figure axes with a line-plot of the
    :meth:`.OBPMetadata.left_over_error_budget`. Below is an example where we plot some ``metadata``
    which exists within our context.

    .. plot::
       :context:
       :nofigs:

       >>> from qiskit_addon_obp.utils.metadata import OBPMetadata
       >>> metadata = OBPMetadata.from_json("docs/_static/dummy_visualization_metadata.json")

    .. plot::
       :alt: Output from the previous code.
       :context: close-figs
       :include-source:

       >>> from matplotlib import pyplot as plt
       >>> from qiskit_addon_obp.utils.visualization import plot_left_over_error_budget
       >>> fig, axes = plt.subplots(1, 1)
       >>> plot_left_over_error_budget(metadata, axes)

    As you can see in the figure above, the number of backpropagated slices is displayed along the
    x-axis. You can think of this as the "time" of the backpropagation algorithm. The left-over
    error budget available at each backpropagation step is displayed along the y-axis. Since each
    observable that was backpropagated has its own budget, they are plotted as separate lines.

    Args:
        metadata: the metadata to be visualized.
        axes: the matplotlib axes in which to plot.
        show_legend: enable/disable showing the legend in the plot.
    """
    for obs_idx in range(len(metadata.backpropagation_history[0].slice_errors)):
        axes.plot(
            range(len(metadata.backpropagation_history) + 1),
            [
                metadata.left_over_error_budget(obs_idx, i)
                for i in range(len(metadata.backpropagation_history) + 1)
            ],
            label=f"Observable #{obs_idx + 1}",
        )
    axes.set_xlabel("backpropagated slice number")
    axes.set_ylabel("left-over error budget")
    _set_legend(axes, show_legend)


def plot_slice_errors(metadata: OBPMetadata, axes: Axes, *, show_legend: bool = True) -> None:
    """Plot the slice errors.

    This method populates the provided figure axes with a bar-plot of the truncation error incurred
    at each backpropagated slice. Below is an example where we plot some ``metadata`` which exists
    within our context.

    .. plot::
       :context:
       :nofigs:

       >>> from qiskit_addon_obp.utils.metadata import OBPMetadata
       >>> metadata = OBPMetadata.from_json("docs/_static/dummy_visualization_metadata.json")

    .. plot::
       :alt: Output from the previous code.
       :context: close-figs
       :include-source:

       >>> from matplotlib import pyplot as plt
       >>> from qiskit_addon_obp.utils.visualization import plot_slice_errors
       >>> fig, axes = plt.subplots(1, 1)
       >>> plot_slice_errors(metadata, axes)

    As you can see in the figure above, the number of backpropagated slices is displayed along the
    x-axis. You can think of this as the "time" of the backpropagation algorithm. The truncation
    error incurred at each backpropagation step is displayed along the y-axis. Since each
    observable is treated individually, they are plotted separately.

    This data is related to the one visualized by :meth:`plot_accumulated_error`. That method will
    plot the cumulative sum of the slice errors along the x-axis.

    Args:
        metadata: the metadata to be visualized.
        axes: the matplotlib axes in which to plot.
        show_legend: enable/disable showing the legend in the plot.
    """
    num_observables = len(metadata.backpropagation_history[0].slice_errors)
    width = 0.8 / num_observables
    offset = -0.4
    for obs_idx in range(num_observables):
        axes.bar(
            [i + offset for i in range(1, len(metadata.backpropagation_history) + 1)],
            [slice_.slice_errors[obs_idx] for slice_ in metadata.backpropagation_history],
            width=width,
            align="edge",
            label=f"Observable #{obs_idx + 1}",
            edgecolor="grey",
        )
        offset += width
    axes.set_xlabel("backpropagated slice number")
    axes.set_ylabel("incurred slice error")
    axes.legend()
    _set_legend(axes, show_legend)


def plot_num_paulis(metadata: OBPMetadata, axes: Axes, *, show_legend: bool = True) -> None:
    """Plot the number of Pauli terms.

    This method populates the provided figure axes with a line-plot of the number of Pauli terms at
    each backpropagated slice. Below is an example where we plot some ``metadata`` which exists
    within our context.

    .. plot::
       :context:
       :nofigs:

       >>> from qiskit_addon_obp.utils.metadata import OBPMetadata
       >>> metadata = OBPMetadata.from_json("docs/_static/dummy_visualization_metadata.json")

    .. plot::
       :alt: Output from the previous code.
       :context: close-figs
       :include-source:

       >>> from matplotlib import pyplot as plt
       >>> from qiskit_addon_obp.utils.visualization import plot_num_paulis
       >>> fig, axes = plt.subplots(1, 1)
       >>> plot_num_paulis(metadata, axes)

    As you can see in the figure above, the number of backpropagated slices is displayed along the
    x-axis. You can think of this as the "time" of the backpropagation algorithm. The number of
    Pauli terms at each backpropagation step is displayed along the y-axis. Since each observable
    is treated individually, they are plotted separately.

    You can also find out the number of unique Pauli terms across all observables by using
    :meth:`plot_sum_paulis`.

    Args:
        metadata: the metadata to be visualized.
        axes: the matplotlib axes in which to plot.
        show_legend: enable/disable showing the legend in the plot.
    """
    for obs_idx in range(len(metadata.backpropagation_history[0].slice_errors)):
        axes.plot(
            range(1, len(metadata.backpropagation_history) + 1),
            [slice_.num_paulis[obs_idx] for slice_ in metadata.backpropagation_history],
            label=f"Observable #{obs_idx + 1}",
        )
    axes.set_xlabel("backpropagated slice number")
    axes.set_ylabel("# Pauli terms")
    _set_legend(axes, show_legend)


def plot_num_truncated_paulis(
    metadata: OBPMetadata, axes: Axes, *, show_legend: bool = True
) -> None:
    """Plot the number of truncated Pauli terms.

    This method populates the provided figure axes with a bar-plot of the number of the truncated
    Pauli terms at each backpropagated slice. Below is an example where we plot some ``metadata``
    which exists within our context.

    .. plot::
       :context:
       :nofigs:

       >>> from qiskit_addon_obp.utils.metadata import OBPMetadata
       >>> metadata = OBPMetadata.from_json("docs/_static/dummy_visualization_metadata.json")

    .. plot::
       :alt: Output from the previous code.
       :context: close-figs
       :include-source:

       >>> from matplotlib import pyplot as plt
       >>> from qiskit_addon_obp.utils.visualization import plot_num_truncated_paulis
       >>> fig, axes = plt.subplots(1, 1)
       >>> plot_num_truncated_paulis(metadata, axes)

    As you can see in the figure above, the number of backpropagated slices is displayed along the
    x-axis. You can think of this as the "time" of the backpropagation algorithm. The number of
    truncated Pauli terms at each backpropagation step is displayed along the y-axis. Since each
    observable is treated individually, they are plotted separately.

    This data can give you additional insight as to how the accumulated error is split across
    multiple Pauli terms (see also the output of :meth:`plot_accumulated_error`).

    Args:
        metadata: the metadata to be visualized.
        axes: the matplotlib axes in which to plot.
        show_legend: enable/disable showing the legend in the plot.
    """
    num_observables = len(metadata.backpropagation_history[0].slice_errors)
    width = 0.8 / num_observables
    offset = -0.4
    for obs_idx in range(num_observables):
        axes.bar(
            [i + offset for i in range(1, len(metadata.backpropagation_history) + 1)],
            [slice_.num_truncated_paulis[obs_idx] for slice_ in metadata.backpropagation_history],
            width=width,
            align="edge",
            label=f"Observable #{obs_idx + 1}",
            edgecolor="grey",
        )
        offset += width
    axes.set_xlabel("backpropagated slice number")
    axes.set_ylabel("# truncated Pauli terms")
    _set_legend(axes, show_legend)


def plot_sum_paulis(metadata: OBPMetadata, axes: Axes, *, show_legend: bool = True) -> None:
    """Plot the total number of all Pauli terms.

    This method populates the provided figure axes with a line-plot of the total number of all Pauli
    terms at each backpropagated slice. Below is an example where we plot some ``metadata`` which
    exists within our context.

    .. plot::
       :context:
       :nofigs:

       >>> from qiskit_addon_obp.utils.metadata import OBPMetadata
       >>> metadata = OBPMetadata.from_json("docs/_static/dummy_visualization_metadata.json")

    .. plot::
       :alt: Output from the previous code.
       :context: close-figs
       :include-source:

       >>> from matplotlib import pyplot as plt
       >>> from qiskit_addon_obp.utils.visualization import plot_sum_paulis
       >>> fig, axes = plt.subplots(1, 1)
       >>> plot_sum_paulis(metadata, axes)

    As you can see in the figure above, the number of backpropagated slices is displayed along the
    x-axis. You can think of this as the "time" of the backpropagation algorithm. The total number
    of all Pauli terms at each backpropagation step is displayed along the y-axis. If
    :attr:`.OBPMetadata.operator_budget.max_paulis` is not `None`, it is displayed as a red horizontal line.

    This data can give you additional insight into how many unique Pauli terms are spread across all
    of the backpropagated observables. See also the output of :meth:`plot_num_paulis` for the
    number of Pauli terms in each observable individually.

    Args:
        metadata: the metadata to be visualized.
        axes: the matplotlib axes in which to plot.
        show_legend: enable/disable showing the legend in the plot.
    """
    if metadata.operator_budget.max_paulis is not None:
        axes.axhline(
            metadata.operator_budget.max_paulis, label="max_paulis", color="red", alpha=0.75
        )
    else:  # pragma: no cover
        pass
    axes.plot(
        range(1, len(metadata.backpropagation_history) + 1),
        [
            slice_.sum_paulis if slice_.sum_paulis is not None else 0
            for slice_ in metadata.backpropagation_history
        ],
        color="black",
    )
    axes.set_xlabel("backpropagated slice number")
    axes.set_ylabel("total # of Pauli terms")
    _set_legend(axes, show_legend)


def plot_num_qwc_groups(metadata: OBPMetadata, axes: Axes, *, show_legend: bool = True) -> None:
    """Plot the number of qubit-wise commuting Pauli groups.

    This method populates the provided figure axes with a line-plot of the number of qubit-wise
    commuting Pauli groups at each backpropagated slice. Below is an example where we plot some
    ``metadata`` which exists within our context.

    .. plot::
       :context:
       :nofigs:

       >>> from qiskit_addon_obp.utils.metadata import OBPMetadata
       >>> metadata = OBPMetadata.from_json("docs/_static/dummy_visualization_metadata.json")

    .. plot::
       :alt: Output from the previous code.
       :context: close-figs
       :include-source:

       >>> from matplotlib import pyplot as plt
       >>> from qiskit_addon_obp.utils.visualization import plot_num_qwc_groups
       >>> fig, axes = plt.subplots(1, 1)
       >>> plot_num_qwc_groups(metadata, axes)

    As you can see in the figure above, the number of backpropagated slices is displayed along the
    x-axis. You can think of this as the "time" of the backpropagation algorithm. The number of
    qubit-wise commuting Pauli groups at each backpropagation step is displayed along the y-axis.
    If :attr:`.OBPMetadata.operator_budget.max_qwc_groups` is not `None`, it is displayed as a red horizontal line.

    Args:
        metadata: the metadata to be visualized.
        axes: the matplotlib axes in which to plot.
        show_legend: enable/disable showing the legend in the plot.
    """
    if metadata.operator_budget.max_qwc_groups is not None:
        axes.axhline(
            metadata.operator_budget.max_qwc_groups, label="max_qwc_groups", color="red", alpha=0.75
        )
    else:  # pragma: no cover
        pass
    axes.plot(
        range(1, len(metadata.backpropagation_history) + 1),
        [
            slice_.num_qwc_groups if slice_.num_qwc_groups is not None else 0
            for slice_ in metadata.backpropagation_history
        ],
        color="black",
    )
    axes.set_xlabel("backpropagated slice number")
    axes.set_ylabel("# of qubit-wise commuting Pauli groups")
    _set_legend(axes, show_legend)


def _set_legend(axes: Axes, show_legend: bool) -> None:
    if show_legend:  # pragma: no cover
        axes.legend()
