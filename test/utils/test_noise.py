# This code is a Qiskit project.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests the PauliLindbladError noise utilities."""

import unittest

from qiskit import QuantumCircuit
from qiskit.quantum_info import PauliList
from qiskit_addon_obp.utils.noise import PauliLindbladErrorInstruction
from qiskit_ibm_runtime.utils.noise_learner_result import PauliLindbladError


class TestPauliLindbladError(unittest.TestCase):
    def test_ple_definition(self):
        err = PauliLindbladError(PauliList(["XX", "XY", "ZX"]), [0.1, 0.2, 0.3])
        inst = PauliLindbladErrorInstruction(err)
        self.assertTrue(isinstance(inst.definition, QuantumCircuit))
        self.assertEqual(inst.definition.data[0].operation.name, "kraus")
