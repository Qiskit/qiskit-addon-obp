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

"""Tests for circuit slicing utility function."""

import unittest

from qiskit import QuantumCircuit
from qiskit_addon_utils.slicing import slice_by_barriers, slice_by_gate_types


class TestSlicingFunctions(unittest.TestCase):
    def test_slice_by_barriers(self):
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.barrier()
        qc.y(1)

        slices = slice_by_barriers(qc)
        self.assertEqual(2, len(slices))
        self.assertEqual(2, slices[0].num_qubits)
        self.assertEqual(2, slices[1].num_qubits)

    def test_slice_by_gate_types(self):
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.y(1)

        slices = slice_by_gate_types(qc)
        self.assertEqual(2, len(slices))
        self.assertEqual(2, slices[0].num_qubits)
        self.assertEqual(2, slices[1].num_qubits)
