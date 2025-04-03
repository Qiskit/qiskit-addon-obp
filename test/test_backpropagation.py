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

"""Tests for operator backpropagation utility functions."""

import sys
import unittest
from math import e
from time import sleep

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit_addon_obp import backpropagate
from qiskit_addon_obp.utils.simplify import OperatorBudget
from qiskit_addon_obp.utils.truncating import setup_budget
from qiskit_addon_utils.slicing import slice_by_barriers, slice_by_depth


class TestBackpropagation(unittest.TestCase):
    def test_backpropagate(self):
        with self.subTest("Single slice"):
            theta = np.pi / 6
            qc_mat = np.array(
                [
                    [np.cos(theta / 2), -np.sin(theta / 2)],
                    [np.sin(theta / 2), np.cos(theta / 2)],
                ]
            )
            X_obs = np.array([[0, 1], [1, 0]])
            target_obs = qc_mat.T @ X_obs @ qc_mat
            qc = QuantumCircuit(1)
            qc.ry(theta, 0)
            obs = SparsePauliOp("X")
            new_obs, new_qc, _ = backpropagate([obs], [qc])

            assert np.allclose(target_obs, new_obs[0].to_matrix())
            self.assertEqual(0, len(new_qc))

            # Backpropagate a 1q operator onto a 2q circ
            qc = QuantumCircuit(2)
            qc.rz(theta, 0)
            qc.ry(theta, 1)

            obs = SparsePauliOp("IX")

            # Should backprop RZ on qubit 0, resulting in IY term, and throw the
            # RY away, as it is outside the light cone of the observable
            new_obs, slices, _ = backpropagate([obs], [qc])
            self.assertEqual({Pauli("IX"), Pauli("IY")}, set(new_obs[0].paulis))
            self.assertEqual(0, len(slices))
            self.assertEqual([], slices)
        with self.subTest("Depth-2"):
            qc = QuantumCircuit(2)
            qc.h(0)
            qc.cx(0, 1)
            obs = SparsePauliOp("ZZ")
            target_obs = SparsePauliOp("ZI")
            new_obs, slices, _ = backpropagate(obs, [qc])
            self.assertEqual(target_obs, new_obs)
            self.assertEqual([], slices)
        with self.subTest("Scattered qargs"):
            qc = QuantumCircuit(5)
            obs = SparsePauliOp("XIYIZ")
            target_obs = SparsePauliOp("XIYIZ")
            new_obs, slices, _ = backpropagate(obs, [qc])

            self.assertEqual(target_obs, new_obs)
            self.assertEqual([], slices)
        with self.subTest("Scattered qargs, ignore op outside light cone"):
            qc = QuantumCircuit(5)
            qc.cx(2, 3)
            obs = SparsePauliOp("ZIIIZ")
            target_obs = SparsePauliOp("ZIIIZ")
            new_obs, new_qc, _ = backpropagate(obs, [qc])

            self.assertEqual(target_obs, new_obs)
            self.assertEqual([], slices)
        with self.subTest("Scattered qargs, add two-qubit op"):
            qc = QuantumCircuit(5)
            qc.cx(2, 4)
            obs = SparsePauliOp("ZIIIZ")
            target_obs = SparsePauliOp("ZIZIZ")
            new_obs, slices, _ = backpropagate(obs, [qc])

            self.assertEqual(target_obs, new_obs)
            self.assertEqual([], slices)
        with self.subTest("Test non-list inputs"):
            theta = np.pi / 6
            qc_mat = np.array(
                [
                    [np.cos(theta / 2), -np.sin(theta / 2)],
                    [np.sin(theta / 2), np.cos(theta / 2)],
                ]
            )
            X_obs = np.array([[0, 1], [1, 0]])
            target_obs = qc_mat.T @ X_obs @ qc_mat
            qc = QuantumCircuit(1)
            qc.ry(theta, 0)
            obs = SparsePauliOp("X")
            new_obs, slices, _ = backpropagate(obs, [qc])

            assert np.allclose(target_obs, new_obs.to_matrix())
            self.assertEqual([], slices)

            qc = QuantumCircuit(2)
            qc.rz(theta, 0)
            qc.ry(theta, 1)

            obs = SparsePauliOp("IX")

            # Should backprop RZ on qubit 0, resulting in IY term, and throw the
            # RY away, as it is outside the light cone of the observable
            new_obs, slices, _ = backpropagate(obs, [qc])
            self.assertEqual({Pauli("IX"), Pauli("IY")}, set(new_obs.paulis))
            self.assertEqual([], slices)
        with self.subTest("multiple slices"):
            theta = np.pi / 6
            Z = np.array([[1, 0], [0, -1]])
            RZ = np.array([[e ** (-1j * (theta / 2)), 0], [0, e ** (1j * (theta / 2))]])
            CX = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])
            obs = np.kron(Z, Z)
            RZRZ = np.kron(RZ, RZ)
            obs_one_slice = CX.T @ obs @ CX
            target_obs = RZRZ.conjugate().T @ obs_one_slice @ RZRZ

            qc = QuantumCircuit(2)
            qc.rz(theta, [0, 1])
            qc.barrier()
            qc.cx(0, 1)
            obs = SparsePauliOp("ZZ")

            # Should backprop entire circuit, no truncation
            slices = slice_by_barriers(qc)
            new_obs, reduced_slices, _ = backpropagate([obs], slices)
            assert np.allclose(target_obs, new_obs[0].to_matrix())
            self.assertEqual([], reduced_slices)
        with self.subTest("max_error_total"):
            theta = np.pi / 6
            qc = QuantumCircuit(1)
            qc.ry(theta, 0)
            obs = SparsePauliOp("X")

            new_obs, new_qc, _ = backpropagate(
                [obs], [qc], truncation_error_budget=setup_budget(max_error_total=0.49)
            )
            self.assertEqual(2, len(new_obs[0].paulis))
            new_obs, slices, _ = backpropagate(
                [obs], [qc], truncation_error_budget=setup_budget(max_error_total=0.5)
            )
            # Z term with coefficient 0.5 should be truncated now
            self.assertEqual(1, len(new_obs[0].paulis))

            self.assertEqual([], slices)
        with self.subTest("max_error_per_slice_as_list"):
            theta = np.pi / 6
            qc1 = QuantumCircuit(1)
            qc2 = QuantumCircuit(1)
            qc1.ry(theta, 0)
            qc2.rz(theta, 0)
            obs = SparsePauliOp("X")

            # No budget at any layer
            new_obs, new_qc, _ = backpropagate(
                obs,
                [qc1, qc2],
                truncation_error_budget=setup_budget(max_error_per_slice=[0.01, 0.01]),
            )
            self.assertEqual(3, len(new_obs.paulis))
            # Can truncate first layer (coeff=0.5), but not second (coeff=0.43)
            new_obs, new_qc, _ = backpropagate(
                obs,
                [qc1, qc2],
                truncation_error_budget=setup_budget(max_error_per_slice=[0.4, 0.5]),
            )
            self.assertEqual(2, len(new_obs.paulis))
            # Can truncate both layers (coeffs = [.43, .5])
            new_obs, new_qc, _ = backpropagate(
                obs,
                [qc1, qc2],
                truncation_error_budget=setup_budget(max_error_per_slice=[0.44, 0.5]),
            )
            self.assertEqual(1, len(new_obs.paulis))
        with self.subTest("barrier"):
            theta = np.pi / 6
            qc = QuantumCircuit(1)
            qc.ry(theta, 0)
            qc.barrier()
            qc.ry(theta, 0)
            obs = SparsePauliOp("X")

            new_obs, reduced_slices, _ = backpropagate([obs], [qc])
            # Z term with coefficient 0.5 should be truncated now
            self.assertEqual([], reduced_slices)
        with self.subTest("reset"):
            qc = QuantumCircuit(1)
            qc.reset(0)
            obs = SparsePauliOp("X")

            new_obs, reduced_slices, _ = backpropagate(obs, [qc])
            # Z term with coefficient 0.5 should be truncated now
            self.assertEqual([], reduced_slices)
            self.assertEqual(SparsePauliOp("I", coeffs=[0.0]), new_obs)
        with self.subTest("simplify"):
            qc = QuantumCircuit(1)
            qc.reset(0)
            obs = SparsePauliOp(["X", "Y", "Z"])
            op_budget_f = OperatorBudget(simplify=False)
            new_obs, _, _ = backpropagate(obs, [qc], operator_budget=op_budget_f)
            self.assertEqual(3, len(new_obs))
            op_budget_t = OperatorBudget(simplify=True)
            new_obs, _, _ = backpropagate(obs, [qc], operator_budget=op_budget_t)
            self.assertEqual(1, len(new_obs))
        with self.subTest("max_error_total + max_error_per_slice"):
            theta = np.pi / 6
            qc = QuantumCircuit(1)
            qc.rz(theta, 0)
            qc.barrier()
            qc.ry(theta, 0)
            obs = SparsePauliOp("X")

            slices = slice_by_barriers(qc)

            # Test max_error_total default
            # Should ignore the Z(coeff .5) term generated from RY and truncate the
            # Y(coeff .43) term generated from RZ
            new_obs, reduced_slices, _ = backpropagate(
                [obs], slices, truncation_error_budget=setup_budget(max_error_per_slice=0.44)
            )
            self.assertEqual({Pauli("X"), Pauli("Z")}, set(new_obs[0].paulis))
            self.assertEqual([], reduced_slices)

            # Test max_error_per_slice default
            # Should ignore the Z(coeff .5) term generated from RY and truncate the
            # Y(coeff .43) term generated from RZ
            new_obs, reduced_slices, _ = backpropagate(
                [obs], slices, truncation_error_budget=setup_budget(max_error_total=0.44)
            )
            self.assertEqual({Pauli("X"), Pauli("Z")}, set(new_obs[0].paulis))
            #            self.assertEqual(reduced_slices)

            # Should truncate both the Z(.5) and Y(.44) terms with .06 remaining budget
            new_obs, reduced_slices, _ = backpropagate(
                [obs],
                slices,
                truncation_error_budget=setup_budget(max_error_total=1, max_error_per_slice=0.5),
            )
            self.assertEqual({Pauli("X")}, set(new_obs[0].paulis))
            self.assertEqual([], reduced_slices)

            # Should not have enough budget to truncate anything past first slice
            new_obs, reduced_slices, _ = backpropagate(
                [obs],
                slices,
                truncation_error_budget=setup_budget(max_error_total=0.8, max_error_per_slice=0.5),
            )
            # Z term with coefficient 0.5 should be truncated now
            self.assertEqual({Pauli("X"), Pauli("Y")}, set(new_obs[0].paulis))
            self.assertEqual([], reduced_slices)

            # Not enough total budget to truncate anything
            new_obs, reduced_slices, _ = backpropagate(
                [obs],
                slices,
                truncation_error_budget=setup_budget(
                    max_error_total=0.001, max_error_per_slice=np.inf
                ),
            )
            self.assertEqual({Pauli("X"), Pauli("Y"), Pauli("Z")}, set(new_obs[0].paulis))
            self.assertEqual([], reduced_slices)

            # Not enough per-slice budget to truncate anything
            new_obs, reduced_slices, _ = backpropagate(
                [obs],
                slices,
                truncation_error_budget=setup_budget(
                    max_error_total=np.inf, max_error_per_slice=0.001
                ),
            )
            self.assertEqual({Pauli("X"), Pauli("Y"), Pauli("Z")}, set(new_obs[0].paulis))
            self.assertEqual([], reduced_slices)
        with self.subTest("slice subsets"):
            theta = np.pi / 6
            qc = QuantumCircuit(1)
            qc.rz(theta, 0)
            qc.barrier()
            qc.ry(theta, 0)
            obs = SparsePauliOp("X")

            slices = slice_by_barriers(qc)

            # Should backprop one slice and stop. It should not truncate the term.
            new_obs, reduced_slices, _ = backpropagate([obs], slices[-1:])
            self.assertEqual({Pauli("X"), Pauli("Z")}, set(new_obs[0].paulis))
            self.assertEqual(0, len(reduced_slices))

            # Should backprop two slices and stop. It should not truncate either term.
            new_obs, reduced_slices, _ = backpropagate([obs], slices[-2:])
            self.assertEqual({Pauli("X"), Pauli("Y"), Pauli("Z")}, set(new_obs[0].paulis))
            self.assertEqual([], reduced_slices)

            # Should backprop two slices and truncate both terms, as per-slice budget is
            # max_error_total / num_slices in this case.
            new_obs, reduced_slices, _ = backpropagate(
                [obs],
                slices[-2:],
                truncation_error_budget=setup_budget(max_error_total=1.0, num_slices=2),
            )
            self.assertEqual({Pauli("X")}, set(new_obs[0].paulis))
            self.assertEqual([], reduced_slices)

            # Should backprop two slices but only truncate the second term, as the default
            # per-slice budget in this case won't allow truncation of the Z term
            new_obs, reduced_slices, _ = backpropagate(
                [obs],
                slices[-2:],
                truncation_error_budget=setup_budget(max_error_total=0.9, num_slices=2),
            )
            self.assertEqual({Pauli("X"), Pauli("Z")}, set(new_obs[0].paulis))
            self.assertEqual([], reduced_slices)
        with self.subTest("L2 budget distribution"):
            theta = np.pi / 6
            qc = QuantumCircuit(1)
            qc.rz(theta, 0)
            qc.barrier()
            qc.ry(theta, 0)
            qc.barrier()
            qc.rz(theta, 0)
            obs = SparsePauliOp("X")

            slices = slice_by_barriers(qc)

            # Need .5 budget to truncate first layer's term with coeff=.5
            new_obs, _, _ = backpropagate(
                [obs],
                slices[-1:],
                truncation_error_budget=setup_budget(
                    max_error_total=0.49,
                    num_slices=1,
                    p_norm=2,
                ),
            )
            self.assertEqual({Pauli("X"), Pauli("Y")}, set(new_obs[0].paulis))
            new_obs, _, _ = backpropagate(
                [obs],
                slices[-1:],
                truncation_error_budget=setup_budget(
                    max_error_total=0.5,
                    num_slices=1,
                    p_norm=2,
                ),
            )
            self.assertEqual({Pauli("X")}, set(new_obs[0].paulis))

            # Need .67 to truncate first two layer's terms with coeffs=[.4330127, .5]
            new_obs, _, _ = backpropagate(
                [obs],
                slices[-2:],
                truncation_error_budget=setup_budget(
                    max_error_total=0.66,
                    num_slices=2,
                    p_norm=2,
                ),
            )
            self.assertEqual({Pauli("X"), Pauli("Y")}, set(new_obs[0].paulis))
            new_obs, _, _ = backpropagate(
                [obs],
                slices[-2:],
                truncation_error_budget=setup_budget(
                    max_error_total=0.67,
                    num_slices=2,
                    p_norm=2,
                ),
            )
            self.assertEqual({Pauli("X")}, set(new_obs[0].paulis))

        with self.subTest("budget distribution for L1 norm"):
            theta = np.pi / 6
            qc = QuantumCircuit(2)
            qc.rz(theta, 0)
            qc.barrier()
            qc.cx(0, 1)
            qc.barrier()
            qc.ry(theta, 0)
            qc.barrier()
            qc = qc.repeat(10).decompose()
            obs = SparsePauliOp("IX")

            slices = slice_by_barriers(qc)

            trunc_budget = setup_budget(max_error_total=0.1, num_slices=30, p_norm=1)

            new_obs, reduced_slices, metadata = backpropagate(
                [obs], slices, truncation_error_budget=trunc_budget
            )

            expected_obs = SparsePauliOp(
                ["XY", "XX", "IY", "IX", "XZ", "IZ"],
                [-0.72261328, 0.08235629, 0.17949574, 0.43425712, 0.20675452, 0.45436588],
            )
            self.assertEqual(new_obs[0], expected_obs)
            self.assertEqual(len(reduced_slices), 0)
            self.assertAlmostEqual(metadata.accumulated_error(0), 0.05558522939071649)

        with self.subTest("budget distribution for L2 norm"):
            theta = np.pi / 6
            qc = QuantumCircuit(2)
            qc.rz(theta, 0)
            qc.barrier()
            qc.cx(0, 1)
            qc.barrier()
            qc.ry(theta, 0)
            qc.barrier()
            qc = qc.repeat(10).decompose()
            obs = SparsePauliOp("IX")

            slices = slice_by_barriers(qc)

            trunc_budget = setup_budget(max_error_total=0.1, num_slices=30, p_norm=2)

            new_obs, reduced_slices, metadata = backpropagate(
                [obs], slices, truncation_error_budget=trunc_budget
            )

            expected_obs = SparsePauliOp(
                ["XY", "XX", "IY", "IX", "XZ", "IZ"],
                [-0.72261328, 0.08235629, 0.1705233, 0.44979783, 0.17567308, 0.45436588],
            )
            self.assertEqual(new_obs[0], expected_obs)
            self.assertEqual(len(reduced_slices), 0)
            self.assertAlmostEqual(metadata.accumulated_error(0), 0.07992961136903981)

        with self.subTest("max_paulis"):
            theta = np.pi / 6
            qc = QuantumCircuit(1)
            qc.rz(theta, 0)
            qc.barrier()
            qc.ry(theta, 0)
            obs = SparsePauliOp("X")

            slices = slice_by_barriers(qc)

            op_budget = OperatorBudget(max_paulis=3)

            # Should backprop entire circuit, no truncation
            new_obs, reduced_slices, _ = backpropagate([obs], slices, operator_budget=op_budget)
            self.assertEqual({Pauli("X"), Pauli("Y"), Pauli("Z")}, set(new_obs[0].paulis))
            self.assertEqual([], reduced_slices)

            # Should backprop one gate, which fills pauli budget
            op_budget = OperatorBudget(max_paulis=2)
            new_obs, reduced_slices, _ = backpropagate([obs], slices, operator_budget=op_budget)
            self.assertEqual({Pauli("X"), Pauli("Z")}, set(new_obs[0].paulis))
            self.assertEqual(1, len(reduced_slices))
            self.assertEqual("rz", reduced_slices[0].data[0].operation.name)

            # Verify the check for a sensible value of max_paulis
            op_budget = OperatorBudget(max_paulis=0)
            with self.assertRaises(ValueError):
                backpropagate(obs, slices, operator_budget=op_budget)

            # Verify the check for already exceeding max_paulis
            op_budget = OperatorBudget(max_paulis=1)
            with self.assertRaises(ValueError):
                backpropagate(SparsePauliOp(["X", "Z"]), slices, operator_budget=op_budget)
        with self.subTest("max_qwc_groups"):
            theta = np.pi / 6
            qc = QuantumCircuit(1)
            qc.rz(theta, 0)
            qc.barrier()
            qc.ry(theta, 0)
            obs = SparsePauliOp("X")

            slices = slice_by_barriers(qc)

            op_budget = OperatorBudget(max_qwc_groups=3)

            # Should backprop entire circuit, no truncation
            new_obs, new_qc, _ = backpropagate([obs], slices, operator_budget=op_budget)
            self.assertEqual({Pauli("X"), Pauli("Y"), Pauli("Z")}, set(new_obs[0].paulis))
            self.assertEqual(0, len(new_qc))

            # Should backprop one gate, which fills pauli budget
            op_budget = OperatorBudget(max_qwc_groups=2)
            new_obs, new_qc, _ = backpropagate([obs], slices, operator_budget=op_budget)
            self.assertEqual({Pauli("X"), Pauli("Z")}, set(new_obs[0].paulis))
            self.assertEqual(1, len(new_qc))

            # Verify the check for a sensible value of max_qwc_groups
            op_budget = OperatorBudget(max_qwc_groups=0)
            with self.assertRaises(ValueError):
                backpropagate(obs, slices, operator_budget=op_budget)

            # Verify the check for already exceeding max_qwc_groups
            op_budget = OperatorBudget(max_qwc_groups=1)
            with self.assertRaises(ValueError):
                backpropagate(SparsePauliOp(["X", "Z"]), slices, operator_budget=op_budget)
        with self.subTest("max_error_total + p_norm=2"):
            qc = QuantumCircuit(2)
            qc.ry(np.pi / 6, 0)
            qc.rz(np.pi / 3, 0)
            qc.cx(0, 1)
            qc.ry(np.pi / 6, 1)
            qc.rz(np.pi / 6, 1)

            obs = SparsePauliOp("IX")

            # No budget for any truncation
            new_obs, _, _ = backpropagate(
                obs, [qc], truncation_error_budget=setup_budget(max_error_total=0.1, p_norm=2)
            )
            self.assertEqual(3, len(new_obs.paulis))
            self.assertEqual(
                {
                    Pauli("XX"),
                    Pauli("XZ"),
                    Pauli("XY"),
                },
                set(new_obs.paulis),
            )
            # Budget to truncate smallest term
            new_obs, _, _ = backpropagate(
                obs, [qc], truncation_error_budget=setup_budget(max_error_total=0.26, p_norm=2)
            )
            self.assertEqual(2, len(new_obs.paulis))
            self.assertEqual(
                {
                    Pauli("XX"),
                    Pauli("XY"),
                },
                set(new_obs.paulis),
            )
            # Budget to truncate 2 smallest terms
            new_obs, _, _ = backpropagate(
                obs, [qc], truncation_error_budget=setup_budget(max_error_total=0.501, p_norm=2)
            )
            self.assertEqual(1, len(new_obs.paulis))
            self.assertEqual(
                {
                    Pauli("XY"),
                },
                set(new_obs.paulis),
            )
        with self.subTest("Mismatching slice widths"):
            obs = SparsePauliOp("X")
            slices = [QuantumCircuit(1), QuantumCircuit(2)]
            with pytest.raises(ValueError) as e_info:
                backpropagate(obs, slices)
            self.assertEqual(
                (
                    "All slices must be defined on the same number of qubits. "
                    "slices[0] contains 1 qubits, but slices[1] contains "
                    "2 qubits."
                ),
                e_info.value.args[0],
            )
        with self.subTest("Mismatching observable size"):
            obs = [SparsePauliOp("X"), SparsePauliOp("XX")]
            slices = [QuantumCircuit(1), QuantumCircuit(1)]
            with pytest.raises(ValueError) as e_info:
                backpropagate(obs, slices)
            self.assertEqual(
                (
                    "Input observables must all act on the same number of qubits. "
                    "observables[0] acts on 1 qubits, but observables[1]"
                    " acts on 2 qubits."
                ),
                e_info.value.args[0],
            )
        with self.subTest("Mismatching slices/observables"):
            obs = [SparsePauliOp("XX"), SparsePauliOp("XX")]
            slices = [QuantumCircuit(1), QuantumCircuit(1)]
            with pytest.raises(ValueError) as e_info:
                backpropagate(obs, slices)
            self.assertEqual(
                (
                    "The input observables must be defined on the same number of "
                    "qubits as the circuit slices."
                ),
                e_info.value.args[0],
            )

    def test_backpropagate_multi(self):
        """Tests back-propagation into multiple observables."""
        with self.subTest("Single slice"):
            theta = np.pi / 6
            qc_mat = np.array(
                [
                    [np.cos(theta / 2), -np.sin(theta / 2)],
                    [np.sin(theta / 2), np.cos(theta / 2)],
                ]
            )
            X_obs = np.array([[0, 1], [1, 0]])
            Z_obs = np.array([[1, 0], [0, -1]])
            target_obs_0 = qc_mat.T @ X_obs @ qc_mat
            target_obs_1 = qc_mat.T @ Z_obs @ qc_mat
            qc = QuantumCircuit(1)
            qc.ry(theta, 0)
            obs = [SparsePauliOp("X"), SparsePauliOp("Z")]
            new_obs, reduced_slices, _ = backpropagate(obs, [qc])

            assert np.allclose(target_obs_0, new_obs[0].to_matrix())
            assert np.allclose(target_obs_1, new_obs[1].to_matrix())
            self.assertEqual([], reduced_slices)

            # Backpropagate a 1q operator onto a 2q circ
            qc = QuantumCircuit(2)
            qc.rz(theta, 0)
            qc.ry(theta, 1)

            obs = [SparsePauliOp("IX"), SparsePauliOp("ZI")]
            new_obs, reduced_slices, _ = backpropagate(obs, [qc])
            # The first observable should backprop RZ on qubit 0, resulting in IY term, and throw
            # the RY away, as it is outside the light cone of the observable
            self.assertEqual({Pauli("IX"), Pauli("IY")}, set(new_obs[0].paulis))
            # The second observable should backprop RY on qubit 1, resulting in YI term, and throw
            # the RZ away, as it is outside the light cone of the observable
            self.assertEqual({Pauli("ZI"), Pauli("XI")}, set(new_obs[1].paulis))
            self.assertEqual([], reduced_slices)
        with self.subTest("multiple slices"):
            theta = np.pi / 6
            Z = np.array([[1, 0], [0, -1]])
            X = np.array([[0, 1], [1, 0]])
            RZ = np.array([[e ** (-1j * (theta / 2)), 0], [0, e ** (1j * (theta / 2))]])
            CX = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])
            obs_zz = np.kron(Z, Z)
            obs_xx = np.kron(X, X)
            RZRZ = np.kron(RZ, RZ)
            obs_zz_one_slice = CX.T @ obs_zz @ CX
            obs_xx_one_slice = CX.T @ obs_xx @ CX
            target_obs_zz = RZRZ.conjugate().T @ obs_zz_one_slice @ RZRZ
            target_obs_xx = RZRZ.conjugate().T @ obs_xx_one_slice @ RZRZ

            qc = QuantumCircuit(2)
            qc.rz(theta, [0, 1])
            qc.barrier()
            qc.cx(0, 1)
            obs = [SparsePauliOp("ZZ"), SparsePauliOp("XX")]

            # Should backprop entire circuit, no truncation
            slices = slice_by_barriers(qc)
            new_obs, reduced_slices, _ = backpropagate(obs, slices)
            self.assertEqual([], reduced_slices)
            assert np.allclose(target_obs_zz, new_obs[0].to_matrix())
            assert np.allclose(target_obs_xx, new_obs[1].to_matrix())
        with self.subTest("max_error_total"):
            theta = np.pi / 6
            qc = QuantumCircuit(1)
            qc.ry(theta, 0)
            obs = [SparsePauliOp("X"), SparsePauliOp("Z")]

            new_obs, new_qc, _ = backpropagate(
                obs, [qc], truncation_error_budget=setup_budget(max_error_total=0.49)
            )
            self.assertEqual(2, len(new_obs[0].paulis))
            self.assertEqual(2, len(new_obs[1].paulis))
            new_obs, slices, _ = backpropagate(
                obs, [qc], truncation_error_budget=setup_budget(max_error_total=0.5)
            )
            # the one term with coefficient 0.5 should be truncated now
            self.assertEqual(1, len(new_obs[0].paulis))
            self.assertEqual(1, len(new_obs[1].paulis))
            self.assertEqual([], slices)
        with self.subTest("max_error_total + max_error_per_slice"):
            theta = np.pi / 6
            qc = QuantumCircuit(1)
            qc.rz(theta, 0)
            qc.barrier()
            qc.ry(theta, 0)
            obs = [SparsePauliOp("X"), SparsePauliOp("Z")]

            slices = slice_by_barriers(qc)

            # Test max_error_total default
            # Should ignore the Z(coeff .5) term generated from RY and truncate the
            # Y(coeff .43) term generated from RZ
            new_obs, reduced_slices, _ = backpropagate(
                obs, slices, truncation_error_budget=setup_budget(max_error_per_slice=0.44)
            )
            self.assertEqual({Pauli("X"), Pauli("Z")}, set(new_obs[0].paulis))
            self.assertEqual({Pauli("Z")}, set(new_obs[1].paulis))
            self.assertEqual([], reduced_slices)

            # Test max_error_per_slice default
            # Should ignore the Z(coeff .5) term generated from RY and truncate the
            # Y(coeff .43) term generated from RZ
            new_obs, reduced_slices, _ = backpropagate(
                obs, slices, truncation_error_budget=setup_budget(max_error_total=0.44)
            )
            self.assertEqual({Pauli("X"), Pauli("Z")}, set(new_obs[0].paulis))
            self.assertEqual({Pauli("X"), Pauli("Z")}, set(new_obs[1].paulis))
            self.assertEqual([], reduced_slices)

            new_obs, reduced_slices, _ = backpropagate(
                obs,
                slices,
                truncation_error_budget=setup_budget(max_error_total=1, max_error_per_slice=0.5),
            )
            # Should truncate both the Z(.5) and Y(.44) terms with .06 remaining budget
            self.assertEqual({Pauli("X")}, set(new_obs[0].paulis))
            # Should truncate both the Y(.25) and X(-.44) terms with .06 remaining budget
            self.assertEqual({Pauli("Z")}, set(new_obs[1].paulis))
            self.assertEqual([], reduced_slices)

            # Should not have enough budget to truncate anything past first slice
            new_obs, reduced_slices, _ = backpropagate(
                obs,
                slices,
                truncation_error_budget=setup_budget(max_error_total=0.8, max_error_per_slice=0.5),
            )
            # Z term with coefficient 0.5 should be truncated now
            self.assertEqual({Pauli("X"), Pauli("Y")}, set(new_obs[0].paulis))
            # Should truncate both the Y(.25) and X(-.44) terms with .06 remaining budget
            self.assertEqual({Pauli("Z")}, set(new_obs[1].paulis))
            self.assertEqual([], reduced_slices)

            # Should not have enough budget to truncate anything past first slice
            new_obs, reduced_slices, _ = backpropagate(
                obs,
                slices,
                truncation_error_budget=setup_budget(max_error_total=0.3, max_error_per_slice=0.3),
            )
            # cannot truncate anything on this observable
            self.assertEqual({Pauli("X"), Pauli("Y"), Pauli("Z")}, set(new_obs[0].paulis))
            # Y term with coefficient 0.25 should be truncated now
            self.assertEqual({Pauli("Z"), Pauli("X")}, set(new_obs[1].paulis))
            self.assertEqual([], reduced_slices)

            # Not enough total budget to truncate anything
            new_obs, reduced_slices, _ = backpropagate(
                obs,
                slices,
                truncation_error_budget=setup_budget(
                    max_error_total=0.001,
                    max_error_per_slice=np.inf,
                ),
            )
            self.assertEqual({Pauli("X"), Pauli("Y"), Pauli("Z")}, set(new_obs[0].paulis))
            self.assertEqual({Pauli("X"), Pauli("Y"), Pauli("Z")}, set(new_obs[1].paulis))
            self.assertEqual([], reduced_slices)

            # Not enough per-slice budget to truncate anything
            new_obs, reduced_slices, _ = backpropagate(
                obs,
                slices,
                truncation_error_budget=setup_budget(
                    max_error_total=np.inf,
                    max_error_per_slice=0.001,
                ),
            )
            self.assertEqual({Pauli("X"), Pauli("Y"), Pauli("Z")}, set(new_obs[0].paulis))
            self.assertEqual({Pauli("X"), Pauli("Y"), Pauli("Z")}, set(new_obs[1].paulis))
            self.assertEqual([], reduced_slices)
        with self.subTest("slice subsets"):
            theta = np.pi / 6
            qc = QuantumCircuit(1)
            qc.rz(theta, 0)
            qc.barrier()
            qc.ry(theta, 0)
            obs = [SparsePauliOp("X"), SparsePauliOp("Z")]

            slices = slice_by_barriers(qc)

            # Should backprop one slice and stop. It should not truncate the term.
            new_obs, new_qc, _ = backpropagate(obs, slices[-1:])
            self.assertEqual({Pauli("X"), Pauli("Z")}, set(new_obs[0].paulis))
            self.assertEqual({Pauli("X"), Pauli("Z")}, set(new_obs[1].paulis))
            self.assertEqual(0, len(new_qc))

            # Should backprop two slices and stop. It should not truncate either term.
            new_obs, reduced_slices, _ = backpropagate(obs, slices[-2:])
            self.assertEqual({Pauli("X"), Pauli("Y"), Pauli("Z")}, set(new_obs[0].paulis))
            self.assertEqual({Pauli("X"), Pauli("Y"), Pauli("Z")}, set(new_obs[1].paulis))
            self.assertEqual([], reduced_slices)

            # Should backprop two slices and truncate both terms, as per-slice budget is
            # max_error_total / max_slices in this case.
            new_obs, reduced_slices, _ = backpropagate(
                obs,
                slices[-2:],
                truncation_error_budget=setup_budget(max_error_total=1.0, num_slices=2),
            )
            self.assertEqual({Pauli("X")}, set(new_obs[0].paulis))
            self.assertEqual({Pauli("Z")}, set(new_obs[1].paulis))
            self.assertEqual([], reduced_slices)

            # Should backprop two slices but only truncate the second term, as the default
            # per-slice budget in this case won't allow truncation of the Z/X term
            new_obs, reduced_slices, _ = backpropagate(
                obs,
                slices[-2:],
                truncation_error_budget=setup_budget(max_error_total=0.9, num_slices=2),
            )
            self.assertEqual({Pauli("X"), Pauli("Z")}, set(new_obs[0].paulis))
            self.assertEqual({Pauli("Z")}, set(new_obs[1].paulis))
            self.assertEqual([], reduced_slices)

            new_obs, reduced_slices, _ = backpropagate(obs, slices[-1:])
            self.assertEqual({Pauli("X"), Pauli("Z")}, set(new_obs[0].paulis))
            self.assertEqual({Pauli("X"), Pauli("Z")}, set(new_obs[1].paulis))
            self.assertEqual(0, len(reduced_slices))
        with self.subTest("max_paulis"):
            theta = np.pi / 6
            qc = QuantumCircuit(1)
            qc.rz(theta, 0)
            qc.barrier()
            qc.ry(theta, 0)
            obs = [SparsePauliOp("X"), SparsePauliOp("Z")]

            slices = slice_by_barriers(qc)

            op_budget = OperatorBudget(max_paulis=3)

            # Should backprop entire circuit, no truncation
            new_obs, reduced_slices, _ = backpropagate(obs, slices, operator_budget=op_budget)
            self.assertEqual({Pauli("X"), Pauli("Y"), Pauli("Z")}, set(new_obs[0].paulis))
            self.assertEqual({Pauli("X"), Pauli("Y"), Pauli("Z")}, set(new_obs[1].paulis))
            self.assertEqual([], reduced_slices)

            # Should backprop one gate, which fills pauli budget
            op_budget = OperatorBudget(max_paulis=2)
            new_obs, reduced_slices, _ = backpropagate(obs, slices, operator_budget=op_budget)
            self.assertEqual({Pauli("X"), Pauli("Z")}, set(new_obs[0].paulis))
            self.assertEqual({Pauli("X"), Pauli("Z")}, set(new_obs[1].paulis))
            self.assertEqual(1, len(reduced_slices))
            self.assertEqual("rz", reduced_slices[0].data[0].operation.name)

            # Verify the check for already exceeding max_paulis
            op_budget = OperatorBudget(max_paulis=1)
            with self.assertRaises(ValueError):
                backpropagate(obs, slices, operator_budget=op_budget)
        with self.subTest("max_qwc_groups"):
            theta = np.pi / 6
            qc = QuantumCircuit(1)
            qc.rz(theta, 0)
            qc.barrier()
            qc.ry(theta, 0)
            obs = [SparsePauliOp("X"), SparsePauliOp("Z")]

            slices = slice_by_barriers(qc)

            op_budget = OperatorBudget(max_qwc_groups=3)

            # Should backprop entire circuit, no truncation
            new_obs, new_qc, _ = backpropagate(obs, slices, operator_budget=op_budget)
            self.assertEqual({Pauli("X"), Pauli("Y"), Pauli("Z")}, set(new_obs[0].paulis))
            self.assertEqual({Pauli("X"), Pauli("Y"), Pauli("Z")}, set(new_obs[1].paulis))
            self.assertEqual(0, len(new_qc))

            # Should backprop one gate, which fills pauli budget
            op_budget = OperatorBudget(max_qwc_groups=2)
            new_obs, new_qc, _ = backpropagate(obs, slices, operator_budget=op_budget)
            self.assertEqual({Pauli("X"), Pauli("Z")}, set(new_obs[0].paulis))
            self.assertEqual({Pauli("X"), Pauli("Z")}, set(new_obs[1].paulis))
            self.assertEqual(1, len(new_qc))

            # Verify the check for already exceeding max_qwc_groups
            op_budget = OperatorBudget(max_qwc_groups=1)
            with self.assertRaises(ValueError):
                backpropagate(obs, slices, operator_budget=op_budget)
        with self.subTest("Correct num_unique_paulis"):
            # This is a regression test against https://github.com/Qiskit/qiskit-addon-obp/issues/21
            qc = QuantumCircuit(2)
            qc.x(0)
            obs = [SparsePauliOp("ZI"), SparsePauliOp(["XX"])]
            new_obs, _, metadata = backpropagate(obs, [qc])
            self.assertEqual(1, len(new_obs[0]))
            self.assertEqual(1, len(new_obs[1]))
            self.assertEqual(1, metadata.backpropagation_history[0].num_unique_paulis[0])
            self.assertEqual(1, metadata.backpropagation_history[0].num_unique_paulis[1])


class TestBackpropagationTimeout:
    def test_backpropagate_timeout(self, subtests, monkeypatch):
        # pylint: disable=no-self-use
        qc = QuantumCircuit(2)
        qc.rx(0.1, 0)
        qc.ry(0.1, 0)
        qc.rz(0.1, 0)
        qc.cx(0, 1)
        slices = slice_by_depth(qc, 1)
        obs = SparsePauliOp("IX")

        timeout_triggered = False

        class MyTimeout(Exception):
            def __init__(self):
                nonlocal timeout_triggered
                timeout_triggered = True

        monkeypatch.setattr("qiskit_addon_obp.backpropagation.TimeoutException", MyTimeout)

        on_win = sys.platform == "win32"
        with subtests.test(msg="Actual timeout"):
            if on_win:
                pytest.skip("Does not run on Windows")
            many_slices = 100_000 * slices

            _, new_qc, _ = backpropagate(obs, many_slices, max_seconds=1)

            with subtests.test(msg="Time should be less than 2 seconds"):
                assert timeout_triggered

            with subtests.test(msg="The resulting circuit should not be empty"):
                assert len(new_qc) > 0

        timeout_triggered = False

        with subtests.test(msg="Reset timeout"):
            if on_win:
                pytest.skip("Does not run on Windows")

            _, new_qc, _ = backpropagate(obs, slices, max_seconds=1)

            with subtests.test(msg="No timeout should have occurred"):
                assert not timeout_triggered

            with subtests.test(msg="The resulting circuit should be empty"):
                assert len(new_qc) == 0

            sleep(1)

        with subtests.test(msg="Handle windows"):
            if not on_win:
                pytest.skip("Only on Windows")
            with pytest.raises(RuntimeError):
                _, _, _ = backpropagate(obs, slices, max_seconds=1)
