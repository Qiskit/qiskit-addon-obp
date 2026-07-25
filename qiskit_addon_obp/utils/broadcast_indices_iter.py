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

import numpy as np


def broadcast_indicies_iter(shape_a: tuple[int, ...], shape_b: tuple[int, ...]):
    len_a = int(np.prod(shape_a))
    len_b = int(np.prod(shape_b))

    shape_c = np.broadcast_shapes(shape_a, shape_b)

    c_broadcast = np.broadcast(
        np.arange(len_a).reshape((len_a, 1)),
        np.arange(len_b).reshape((1, len_b)),
    )
    for _a_idx, _b_idx in c_broadcast:
        a_idx = np.unravel_index(_a_idx, shape_a)
        b_idx = np.unravel_index(_b_idx, shape_b)
        c_idx = np.unravel_index(c_broadcast.index - 1, shape_c)

        yield (a_idx, b_idx, c_idx)
