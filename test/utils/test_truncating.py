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

"""Tests for the ``obp.utils.truncating`` module."""

import unittest

import numpy as np
from qiskit_addon_obp.utils.truncating import TruncationErrorBudget, setup_budget


class TestTruncating(unittest.TestCase):
    def test_setup_budget_fp_meps(self):
        target = TruncationErrorBudget(per_slice_budget=[0.1], max_error_total=1.0, p_norm=1)
        budget = setup_budget(max_error_per_slice=0.1, max_error_total=1.0)
        self.assertEqual(target, budget)

    def test_setup_budget_num_slices(self):
        target = TruncationErrorBudget(per_slice_budget=[0.2], max_error_total=1.0, p_norm=1)
        budget = setup_budget(max_error_total=1.0, num_slices=5)
        self.assertEqual(target, budget)

    def test_setup_budget_num_slices_pnorm(self):
        max_error_total = 1.0
        num_slices = 5
        p_norm = 2
        per_slice_budget = [max_error_total / num_slices]
        target = TruncationErrorBudget(
            per_slice_budget=per_slice_budget, max_error_total=max_error_total, p_norm=p_norm
        )
        budget = setup_budget(max_error_total=max_error_total, num_slices=num_slices, p_norm=p_norm)
        self.assertEqual(target, budget)

    def test_setup_budget_all_none(self):
        """Tests the setup_budget method."""
        with (
            self.subTest("ValueError on max_error_per_slice and max_error_total being None"),
            self.assertRaises(ValueError),
        ):
            _ = setup_budget(max_error_per_slice=None, max_error_total=None)

    def test_setup_budget_no_max(self):
        """Test the `setup_budget` method without `max_error_total`."""
        target = TruncationErrorBudget(per_slice_budget=[0.1], max_error_total=np.inf)
        budget = setup_budget(max_error_per_slice=0.1, max_error_total=None)
        self.assertEqual(target, budget)

    def test_invalid_setup_budget(self):
        """Test the `setup_budget` method with empty inputs."""
        with self.assertRaises(ValueError):
            _ = setup_budget(max_error_per_slice=None, max_error_total=None)
