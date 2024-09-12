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

"""Tests for the operator simplification utility function."""

import unittest

import numpy as np
from qiskit.circuit import Parameter
from qiskit.quantum_info import Operator, SparsePauliOp
from qiskit_addon_obp.utils.simplify import simplify


class TestSimplify(unittest.TestCase):
    def test_simplify(self):
        """Test simplify method"""
        coeffs = [3 + 1j, -3 - 1j, 0, 4, -5, 2.2, -1.1j]
        labels = ["IXI", "IXI", "ZZZ", "III", "III", "XXX", "XXX"]
        spp_op = SparsePauliOp.from_list(zip(labels, coeffs))
        simplified_op, metadata = simplify(spp_op)
        target_coeffs = [-1, 2.2 - 1.1j]
        target_labels = ["III", "XXX"]
        target_op = SparsePauliOp.from_list(zip(target_labels, target_coeffs))
        with self.subTest("Assert Operator"):
            self.assertEqual(simplified_op, target_op)
            np.testing.assert_array_equal(simplified_op.paulis.phase, np.zeros(simplified_op.size))
        with self.subTest("Assert Metadata"):
            self.assertEqual(metadata.num_unique_paulis, 3)
            self.assertEqual(metadata.num_duplicate_paulis, 4)
            self.assertEqual(metadata.num_trimmed_paulis, 5)
            self.assertEqual(metadata.sum_trimmed_coeffs, 0)

    def test_simplify_zero(self):
        """Test simplify method for {num_qubits} qubits with zero operators."""
        coeffs = [3 + 1j, -3 - 1j, 0, 4, -5, 2.2, -1.1j]
        labels = ["IXI", "IXI", "ZZZ", "III", "III", "XXX", "XXX"]
        spp_op = SparsePauliOp.from_list(zip(labels, coeffs))
        zero_op = spp_op - spp_op
        simplified_op, metadata = simplify(zero_op)
        value = Operator(simplified_op)
        target = Operator(zero_op)
        with self.subTest("Assert Operator"):
            self.assertEqual(value, target)
            np.testing.assert_array_equal(simplified_op.coeffs, [0])
            np.testing.assert_array_equal(zero_op.paulis.phase, np.zeros(zero_op.size))
            np.testing.assert_array_equal(simplified_op.paulis.phase, np.zeros(simplified_op.size))
        with self.subTest("Assert Metadata"):
            self.assertEqual(metadata.num_unique_paulis, 3)
            self.assertEqual(metadata.num_duplicate_paulis, 11)
            self.assertEqual(metadata.num_trimmed_paulis, 14)
            self.assertEqual(metadata.sum_trimmed_coeffs, 0)

    def test_simplify_parameters(self):
        """Test simplify methods for parameterized SparsePauliOp."""
        a = Parameter("a")
        coeffs = np.array([a, -a, 0, a, a, a, 2 * a])
        labels = ["IXI", "IXI", "ZZZ", "III", "III", "XXX", "XXX"]
        spp_op = SparsePauliOp(labels, coeffs)
        simplified_op, metadata = simplify(spp_op)
        target_coeffs = np.array([2 * a, 3 * a])
        target_labels = ["III", "XXX"]
        target_op = SparsePauliOp(target_labels, target_coeffs)
        with self.subTest("Assert Operator"):
            self.assertEqual(simplified_op, target_op)
            np.testing.assert_array_equal(simplified_op.paulis.phase, np.zeros(simplified_op.size))
        with self.subTest("Assert Metadata"):
            self.assertEqual(metadata.num_unique_paulis, 3)
            self.assertEqual(metadata.num_duplicate_paulis, 4)
            self.assertEqual(metadata.num_trimmed_paulis, 5)
            self.assertEqual(metadata.sum_trimmed_coeffs, 0)

    def test_simplify_no_op(self):
        """Test simplify method on an operator without effect."""
        coeffs = [3, 2]
        labels = ["IXI", "ZZZ"]
        spp_op = SparsePauliOp.from_list(zip(labels, coeffs))
        simplified_op, metadata = simplify(spp_op)
        with self.subTest("Assert Operator"):
            self.assertEqual(simplified_op, spp_op)
            np.testing.assert_array_equal(simplified_op.paulis.phase, np.zeros(simplified_op.size))
        with self.subTest("Assert Metadata"):
            self.assertEqual(metadata.num_unique_paulis, 2)
            self.assertEqual(metadata.num_duplicate_paulis, 0)
            self.assertEqual(metadata.num_trimmed_paulis, 0)
            self.assertEqual(metadata.sum_trimmed_coeffs, 0)

    def test_simplify_tolerances(self):
        """Test simplify method with custom tolerances"""
        coeffs = [1.0, 0.1, 0.01, 0.001]
        labels = ["III", "ZII", "IZI", "IIZ"]
        spp_op = SparsePauliOp.from_list(zip(labels, coeffs))
        simplified_op, metadata = simplify(spp_op, atol=1e-2, rtol=1e-2)
        target_coeffs = [1.0, 0.1]
        target_labels = ["III", "ZII"]
        target_op = SparsePauliOp.from_list(zip(target_labels, target_coeffs))
        with self.subTest("Assert Operator"):
            self.assertEqual(simplified_op, target_op)
            np.testing.assert_array_equal(simplified_op.paulis.phase, np.zeros(simplified_op.size))
        with self.subTest("Assert Metadata"):
            self.assertEqual(metadata.num_unique_paulis, 2)
            # FIXME: the following fails due to a known bug. The `num_duplicate_paulis` cannot
            # differentiate between actual duplicates and values which have been trimmed due to the
            # chosen tolerances.
            # self.assertEqual(metadata.num_duplicate_paulis, 0)
            self.assertEqual(metadata.num_trimmed_paulis, 2)
            self.assertEqual(metadata.sum_trimmed_coeffs, 0)
