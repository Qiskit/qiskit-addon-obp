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

"""Tests for the OBP Metadata classes."""

from __future__ import annotations

import sys
import tempfile
import unittest

from qiskit_addon_obp.utils.metadata import OBPMetadata, SliceMetadata
from qiskit_addon_obp.utils.simplify import OperatorBudget
from qiskit_addon_obp.utils.truncating import TruncationErrorBudget


class TestOBPMetadata(unittest.TestCase):
    def setUp(self) -> None:
        self.expected = OBPMetadata(
            truncation_error_budget=TruncationErrorBudget([0.001], 0.01, 1, 1e-6),
            num_slices=None,
            operator_budget=OperatorBudget(50, 12, True, 1e-10, 1e-10),
            backpropagation_history=[
                SliceMetadata(
                    slice_errors=[0.0, 0.0],
                    raw_num_paulis=[0, 0],
                    num_unique_paulis=None,
                    num_duplicate_paulis=None,
                    num_trimmed_paulis=None,
                    sum_trimmed_coeffs=None,
                    num_truncated_paulis=[0, 0],
                    num_paulis=[2, 2],
                    sum_paulis=2,
                    num_qwc_groups=2,
                ),
                SliceMetadata(
                    slice_errors=[0.0, 0.0],
                    raw_num_paulis=[0, 0],
                    num_unique_paulis=None,
                    num_duplicate_paulis=None,
                    num_trimmed_paulis=None,
                    sum_trimmed_coeffs=None,
                    num_truncated_paulis=[0, 0],
                    num_paulis=[3, 3],
                    sum_paulis=3,
                    num_qwc_groups=3,
                ),
                SliceMetadata(
                    slice_errors=[0.0028387171342200146, 0.0028377078723825274],
                    raw_num_paulis=[0, 0],
                    num_unique_paulis=None,
                    num_duplicate_paulis=None,
                    num_trimmed_paulis=None,
                    sum_trimmed_coeffs=None,
                    num_truncated_paulis=[1, 1],
                    num_paulis=[4, 4],
                    sum_paulis=5,
                    num_qwc_groups=3,
                ),
                SliceMetadata(
                    slice_errors=[0.000706650599441302, 0.0007069019274547999],
                    raw_num_paulis=[0, 0],
                    num_unique_paulis=None,
                    num_duplicate_paulis=None,
                    num_trimmed_paulis=None,
                    sum_trimmed_coeffs=None,
                    num_truncated_paulis=[1, 1],
                    num_paulis=[6, 6],
                    sum_paulis=7,
                    num_qwc_groups=3,
                ),
                SliceMetadata(
                    slice_errors=[0.0003898343375198982, 0.0004956955290468059],
                    raw_num_paulis=[0, 0],
                    num_unique_paulis=None,
                    num_duplicate_paulis=None,
                    num_trimmed_paulis=None,
                    sum_trimmed_coeffs=None,
                    num_truncated_paulis=[3, 3],
                    num_paulis=[10, 9],
                    sum_paulis=13,
                    num_qwc_groups=5,
                ),
                SliceMetadata(
                    slice_errors=[0.0003528021807868488, 0.0009668269109679079],
                    raw_num_paulis=[0, 0],
                    num_unique_paulis=None,
                    num_duplicate_paulis=None,
                    num_trimmed_paulis=None,
                    sum_trimmed_coeffs=None,
                    num_truncated_paulis=[3, 4],
                    num_paulis=[15, 12],
                    sum_paulis=20,
                    num_qwc_groups=7,
                ),
                SliceMetadata(
                    slice_errors=[0.0025508975511764273, 0.0016109709987377695],
                    raw_num_paulis=[0, 0],
                    num_unique_paulis=None,
                    num_duplicate_paulis=None,
                    num_trimmed_paulis=None,
                    sum_trimmed_coeffs=None,
                    num_truncated_paulis=[26, 16],
                    num_paulis=[21, 15],
                    sum_paulis=25,
                    num_qwc_groups=10,
                ),
                SliceMetadata(
                    slice_errors=[0.0006514064295348555, 0.0009756328394833543],
                    raw_num_paulis=[0, 0],
                    num_unique_paulis=None,
                    num_duplicate_paulis=None,
                    num_trimmed_paulis=None,
                    sum_trimmed_coeffs=None,
                    num_truncated_paulis=[15, 16],
                    num_paulis=[23, 19],
                    sum_paulis=31,
                    num_qwc_groups=10,
                ),
                SliceMetadata(
                    slice_errors=[0.001041962447804151, 0.000787486904529152],
                    raw_num_paulis=[0, 0],
                    num_unique_paulis=None,
                    num_duplicate_paulis=None,
                    num_trimmed_paulis=None,
                    sum_trimmed_coeffs=None,
                    num_truncated_paulis=[16, 12],
                    num_paulis=[25, 20],
                    sum_paulis=34,
                    num_qwc_groups=10,
                ),
                SliceMetadata(
                    slice_errors=[0.0012105923449034592, 0.0015736014926067718],
                    raw_num_paulis=[0, 0],
                    num_unique_paulis=None,
                    num_duplicate_paulis=None,
                    num_trimmed_paulis=None,
                    sum_trimmed_coeffs=None,
                    num_truncated_paulis=[17, 13],
                    num_paulis=[26, 19],
                    sum_paulis=34,
                    num_qwc_groups=10,
                ),
                SliceMetadata(
                    slice_errors=[0.0002044007557692882, 1.564389619523384e-05],
                    raw_num_paulis=[0, 0],
                    num_unique_paulis=None,
                    num_duplicate_paulis=None,
                    num_trimmed_paulis=None,
                    sum_trimmed_coeffs=None,
                    num_truncated_paulis=[4, 3],
                    num_paulis=[35, 30],
                    sum_paulis=46,
                    num_qwc_groups=16,
                ),
            ],
            num_backpropagated_slices=10,
        )

    def test_from_json(self):
        metadata = OBPMetadata.from_json("docs/_static/dummy_visualization_metadata.json")
        self.assertEqual(metadata, self.expected)

    def test_to_json(self):
        with tempfile.NamedTemporaryFile(delete=sys.platform != "win32") as file:
            self.expected.to_json(file.name, indent=2)

            with (
                open(file.name) as reality,
                open("docs/_static/dummy_visualization_metadata.json") as truth,
            ):
                self.assertEqual(reality.read(), truth.read())

    def test_accumulated_error(self):
        with self.subTest("observable 1"):
            expected_obs_1 = [
                0.0,
                0.0,
                0.0,
                0.0028387171342200146,
                0.0035453677336613164,
                0.003935202071181214,
                0.004288004251968063,
                0.006838901803144491,
                0.0074903082326793465,
                0.008532270680483497,
                0.009742863025386957,
            ]
            reality_obs_1 = [
                self.expected.accumulated_error(0, i)
                for i in range(len(self.expected.backpropagation_history))
            ]
            for e, r in zip(expected_obs_1, reality_obs_1):
                self.assertAlmostEqual(e, r, places=10)
        with self.subTest("observable 2"):
            expected_obs_2 = [
                0.0,
                0.0,
                0.0,
                0.0028377078723825274,
                0.0035446097998373272,
                0.004040305328884133,
                0.00500713223985204,
                0.0066181032385898095,
                0.007593736078073164,
                0.008381222982602316,
                0.009954824475209088,
            ]
            reality_obs_2 = [
                self.expected.accumulated_error(1, i)
                for i in range(len(self.expected.backpropagation_history))
            ]
            for e, r in zip(expected_obs_2, reality_obs_2):
                self.assertAlmostEqual(e, r, places=10)

    def test_left_over_error_budget(self):
        with self.subTest("observable 1"):
            expected_obs_1 = [
                0.001,
                0.002,
                0.003,
                0.0011612828657799855,
                0.0014546322663386835,
                0.0020647979288187855,
                0.0027119957480319368,
                0.0011610981968555095,
                0.0015096917673206541,
                0.0014677293195165027,
                0.00025713697461304287,
            ]
            reality_obs_1 = [
                self.expected.left_over_error_budget(0, i)
                for i in range(len(self.expected.backpropagation_history))
            ]
            for e, r in zip(expected_obs_1, reality_obs_1):
                self.assertAlmostEqual(e, r, places=10)
        with self.subTest("observable 2"):
            expected_obs_2 = [
                0.001,
                0.002,
                0.003,
                0.0011622921276174727,
                0.0014553902001626729,
                0.001959694671115867,
                0.001992867760147959,
                0.0013818967614101895,
                0.0014062639219268355,
                0.0016187770173976836,
                4.5175524790911845e-05,
            ]
            reality_obs_2 = [
                self.expected.left_over_error_budget(1, i)
                for i in range(len(self.expected.backpropagation_history))
            ]
            for e, r in zip(expected_obs_2, reality_obs_2):
                self.assertAlmostEqual(e, r, places=10)
