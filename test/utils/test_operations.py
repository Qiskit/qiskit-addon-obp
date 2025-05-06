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

import unittest
from functools import reduce

import numpy as np
import pytest
from qiskit.circuit.library import CXGate, XGate
from qiskit.quantum_info import PauliList, SparsePauliOp
from qiskit_addon_obp.utils.operations import (
    apply_op_to,
    apply_ple_to,
    apply_reset_to,
    reduce_op,
    to_global_op,
)
from qiskit_ibm_runtime.utils.noise_learner_result import PauliLindbladError


class TestOperationsFunctions(unittest.TestCase):
    def setUp(self):
        self.op = SparsePauliOp("XYZ")
        self.X_mat = np.fliplr(np.identity(2))
        self.I_mat = np.identity(2)

    def test_apply_op_to(self):
        with self.subTest("Basic test. Same qubits."):
            # CXdag @ IX @ CX == XX
            target_op = np.fliplr(np.identity(4))
            obs = SparsePauliOp("IX")
            op = SparsePauliOp.from_operator(CXGate())
            qargs = [0, 1]

            new_op, new_qargs = apply_op_to(obs, qargs, op, qargs, apply_as_transform=True)

            assert np.all(new_op.to_matrix() == target_op)
            assert np.all(qargs == new_qargs)
        with self.subTest("Basic test. Left apply."):
            # ZI @ IX == ZX
            target_op = np.array(
                [
                    [0.0, 1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, -1.0],
                    [0.0, 0.0, -1.0, 0.0],
                ]
            )
            obs = SparsePauliOp("IX")
            op = SparsePauliOp("ZI")
            qargs = [0, 1]

            new_op, new_qargs = apply_op_to(obs, qargs, op, qargs)

            assert np.all(new_op.to_matrix() == target_op)
            assert np.all(qargs == new_qargs)
        with self.subTest("Right-add XGate"):
            # Manually calculate expected output operator and qargs
            X_mat = self.X_mat
            I_mat = self.I_mat
            expanded_x = reduce(np.kron, [X_mat, I_mat, I_mat])
            expanded_obs = reduce(np.kron, [I_mat, I_mat, X_mat])
            target_op = expanded_x.T @ expanded_obs @ expanded_x
            target_qargs = [0, 1, 2]

            # Use apply_op_to with Qiskit inputs
            obs = SparsePauliOp("IX")
            op = SparsePauliOp.from_operator(XGate())
            new_op, new_qargs = apply_op_to(obs, [0, 1], op, [2], apply_as_transform=True)

            # Compare
            assert np.all(target_op == new_op.to_matrix())
            assert np.all(target_qargs == new_qargs)
        with self.subTest("Left-add XGate, non-overlapping qubit"):
            # Manually calculate expected output operator and qargs
            X_mat = self.X_mat
            I_mat = self.I_mat
            expanded_obs = reduce(np.kron, [I_mat, I_mat, X_mat])
            expanded_x = reduce(np.kron, [X_mat, I_mat, I_mat])
            target_op = expanded_x.T @ expanded_obs @ expanded_x
            target_qargs = [0, 1, 2]

            # Use apply_op_to with Qiskit inputs
            obs = SparsePauliOp("IX")
            op = SparsePauliOp.from_operator(XGate())
            new_op, new_qargs = apply_op_to(obs, [0, 1], op, [2], apply_as_transform=True)

            # Compare
            assert np.all(target_op == new_op.to_matrix())
            assert np.all(target_qargs == new_qargs)
        with self.subTest("Left-add CXGate, partially overlapping qubits"):
            # Manually calculate expected output operator and qargs
            CX_mat = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])
            X_mat = self.X_mat
            I_mat = self.I_mat
            expanded_obs = reduce(np.kron, [I_mat, X_mat, X_mat])
            expanded_cx = reduce(np.kron, [CX_mat, I_mat])
            target_op = expanded_cx.T @ expanded_obs @ expanded_cx
            target_qargs = [0, 1, 2]

            # Use apply_op_to with Qiskit inputs
            obs = SparsePauliOp("XX")
            op = SparsePauliOp.from_operator(CXGate())
            new_op, new_qargs = apply_op_to(obs, [0, 1], op, [1, 2], apply_as_transform=True)

            # Compare
            assert np.all(target_op == new_op.to_matrix())
            assert np.all(target_qargs == new_qargs)
        with self.subTest("Left-add CXGate, non-overlapping, scattered qubits"):
            # Manually calculate expected output operator and qargs
            CX_mat = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])
            X_mat = self.X_mat
            I_mat = self.I_mat
            expanded_obs = reduce(np.kron, [I_mat, I_mat, X_mat, X_mat])
            expanded_cx = reduce(np.kron, [CX_mat, I_mat, I_mat])
            target_op = expanded_cx.T @ expanded_obs @ expanded_cx
            target_qargs = [2, 4, 6, 8]

            # Use apply_op_to with Qiskit inputs
            obs = SparsePauliOp("XX")
            op = SparsePauliOp.from_operator(CXGate())
            new_op, new_qargs = apply_op_to(obs, [2, 4], op, [6, 8], apply_as_transform=True)

            # Compare
            assert np.all(target_op == new_op.to_matrix())
            assert np.all(target_qargs == new_qargs)
        with self.subTest("Wrong number of qargs."):
            with pytest.raises(ValueError) as e_info:
                apply_op_to(self.op, [1, 2], XGate(), [1], apply_as_transform=True)
            self.assertEqual(
                "The number of qubits in the operator (3) does not match the number of qargs (2).",
                e_info.value.args[0],
            )
            with pytest.raises(ValueError) as e_info:
                apply_op_to(self.op, [1, 2, 3], SparsePauliOp.from_operator(XGate()), [1, 2])
            self.assertEqual(
                "The number of qubits in the operator (1) does not match the number of qargs (2).",
                e_info.value.args[0],
            )

    def test_to_global_op(self):
        with self.subTest("Basic test."):
            target_op = SparsePauliOp("IIXYZIII")
            global_op = to_global_op(self.op, [3, 4, 5], 8)
            self.assertEqual(target_op, global_op)
        with self.subTest("Non contiguous qubits."):
            target_op = SparsePauliOp("IIXIIYZI")
            global_op = to_global_op(self.op, [1, 2, 5], 8)
        with self.subTest("qarg out of range."):
            with pytest.raises(ValueError) as e_info:
                to_global_op(self.op, [6, 7, 8], 8)
            self.assertEqual(
                "qargs contains qubit ID (8), but the global system contains only (8) qubits.",
                e_info.value.args[0],
            )
        with self.subTest("Negative qubit ID."):
            with pytest.raises(ValueError) as e_info:
                to_global_op(self.op, [-3, 4, 5], 8)
            self.assertEqual(
                "qargs may not contain a negative qubit ID. Found: (-3).",
                e_info.value.args[0],
            )

    def test_reduce_op(self):
        with self.subTest("Single qubit."):
            target_op = SparsePauliOp("X", coeffs=np.array([2.0]))
            target_qargs = [0]
            reduced_op, qargs = reduce_op(SparsePauliOp("X", coeffs=np.array([2.0])))
            self.assertEqual(target_op, reduced_op)
            self.assertEqual(target_qargs, qargs)
        with self.subTest("Basic test."):
            target_op = SparsePauliOp("XYZ", coeffs=np.array([2.0]))
            target_qargs = [1, 2, 3]
            reduced_op, qargs = reduce_op(SparsePauliOp("IIXYZI", coeffs=np.array([2.0])))
            self.assertEqual(target_op, reduced_op)
            self.assertEqual(target_qargs, qargs)
        with self.subTest("Multiple terms."):
            input_op = SparsePauliOp(
                ["ZIIZIXI", "XXIIIZY", "IIIXIIX"], coeffs=np.array([2.0, 4.0, 6.0])
            )
            target_op = SparsePauliOp(["ZIZXI", "XXIZY", "IIXIX"], coeffs=np.array([2.0, 4.0, 6.0]))
            target_qargs = [0, 1, 3, 5, 6]
            reduced_op, qargs = reduce_op(input_op)
            self.assertEqual(target_op, reduced_op)
            self.assertEqual(target_qargs, qargs)
        with self.subTest("Identity."):
            with pytest.raises(ValueError) as e_info:
                reduce_op(SparsePauliOp("III"))
            self.assertEqual(
                "Input operator may not be the identity operator.",
                e_info.value.args[0],
            )
            with pytest.raises(ValueError) as e_info:
                reduce_op(SparsePauliOp(["III"]))
            self.assertEqual(
                "Input operator may not be the identity operator.",
                e_info.value.args[0],
            )

    def test_apply_reset_to(self):
        with self.subTest("Basic test."):
            op = SparsePauliOp(["XYZ", "YZX", "ZXY"], coeffs=[1.0, 2.0, 3.0])
            target_op = SparsePauliOp(["XYI"], coeffs=[1.0])
            qubit_id = 0
            new_op = apply_reset_to(op, qubit_id)
            new_op = new_op.simplify()
            self.assertEqual(target_op, new_op)

            target_op = SparsePauliOp(["YIX"], coeffs=[2.0])
            qubit_id = 1
            new_op = apply_reset_to(op, qubit_id)
            new_op = new_op.simplify()
            self.assertEqual(target_op, new_op)

            target_op = SparsePauliOp(["IXY"], coeffs=[3.0])
            qubit_id = 2
            new_op = apply_reset_to(op, qubit_id)
            new_op = new_op.simplify()
            self.assertEqual(target_op, new_op)

        with self.subTest("Inplace"):
            op = SparsePauliOp(["XYZ", "YZX", "ZXY"], coeffs=[1.0, 2.0, 3.0])
            target_op = SparsePauliOp(["XYI"], coeffs=[1.0])
            qubit_id = 0
            new_op = apply_reset_to(op, qubit_id, inplace=True)
            self.assertEqual(op, new_op)
            new_op = new_op.simplify()
            self.assertEqual(target_op, new_op)

    def test_apply_ple_to(self):
        with self.subTest("same support on 1 qubit"):
            op = SparsePauliOp(["Z"], [1.0])
            lerr = PauliLindbladError(PauliList(["X"]), [1e-3])
            target_op = SparsePauliOp(["Z"], [0.998002])
            new_op, new_qargs = apply_ple_to(op, [0], lerr, [0])
            self.assertEqual(new_op, target_op)
            self.assertEqual(new_qargs, [0])

        with self.subTest("same support on 2 qubits"):
            op = SparsePauliOp(["ZX"], [1.0])
            lerr = PauliLindbladError(PauliList(["XX"]), [1e-3])
            target_op = SparsePauliOp(["ZX"], [0.998002])
            new_op, new_qargs = apply_ple_to(op, [0, 1], lerr, [0, 1])
            self.assertEqual(new_op, target_op)
            self.assertEqual(new_qargs, [0, 1])

        with self.subTest("disjoint support"):
            op = SparsePauliOp(["Z"], [1.0])
            lerr = PauliLindbladError(PauliList(["X"]), [1e-3])
            target_op = SparsePauliOp(["ZI"], [1.0])
            new_op, new_qargs = apply_ple_to(op, [1], lerr, [0])
            self.assertEqual(new_op, target_op)
            self.assertEqual(new_qargs, [0, 1])

        with self.subTest("overlapping support"):
            op = SparsePauliOp(["XZ"], [1.0])
            lerr = PauliLindbladError(PauliList(["XX"]), [1e-3])
            target_op = SparsePauliOp(["XZI"], [0.998002])
            new_op, new_qargs = apply_ple_to(op, [1, 2], lerr, [0, 1])
            self.assertEqual(new_op, target_op)
            self.assertEqual(new_qargs, [0, 1, 2])

        with self.subTest("multiple PLE generators"):
            op = SparsePauliOp(["XZ"], [1.0])
            lerr = PauliLindbladError(PauliList(["XX", "YY"]), [1e-3, 1e-2])
            target_op = SparsePauliOp(["XZI"], [0.97824024])
            new_op, new_qargs = apply_ple_to(op, [1, 2], lerr, [0, 1])
            self.assertEqual(new_op, target_op)
            self.assertEqual(new_qargs, [0, 1, 2])

        with self.subTest("multiple terms in observable"):
            op = SparsePauliOp(["XZ", "YY"], [1.0])
            lerr = PauliLindbladError(PauliList(["YY"]), [1e-2])
            target_op = SparsePauliOp(["XZI", "YYI"], [0.98019867, 1.0])
            new_op, new_qargs = apply_ple_to(op, [1, 2], lerr, [0, 1])
            self.assertEqual(new_op, target_op)
            self.assertEqual(new_qargs, [0, 1, 2])

        with self.subTest("multiple terms in observable and multiple PLE generators"):
            op = SparsePauliOp(["XZ", "YY"], [1.0])
            lerr = PauliLindbladError(PauliList(["XX", "YY"]), [1e-3, 1e-2])
            target_op = SparsePauliOp(["XZI", "YYI"], [0.97824024, 0.998002])
            new_op, new_qargs = apply_ple_to(op, [1, 2], lerr, [0, 1])
            self.assertEqual(new_op, target_op)
            self.assertEqual(new_qargs, [0, 1, 2])
