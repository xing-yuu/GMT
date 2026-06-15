import unittest

import jittor as jt
import numpy as np

from train._utils import build_dof_indices, ebe_matvec, scatter_add


class EBEMatvecTest(unittest.TestCase):
    def setUp(self):
        jt.flags.use_cuda = int(jt.has_cuda)

    def test_scatter_add_repeated_indices_and_gradient(self):
        src = jt.array([[1.0], [2.0], [3.0], [4.0]])
        index = jt.array([0, 0, 1, 0]).int32()

        output = scatter_add(src, index, 2)
        gradient = jt.grad((output * output).sum(), src)

        np.testing.assert_allclose(output.numpy(), [[7.0], [3.0]])
        np.testing.assert_allclose(gradient.numpy(), [[14.0], [14.0], [6.0], [14.0]])

    def test_multiple_elements_and_right_hand_sides(self):
        ke = np.arange(24 * 24, dtype=np.float32).reshape(24, 24) / 100.0
        node_index = np.stack(
            [np.arange(8, dtype=np.int32), np.arange(8, dtype=np.int32)]
        )
        u = np.arange(8 * 3 * 2, dtype=np.float32).reshape(8, 3, 2) / 10.0

        actual = ebe_matvec(
            jt.array(ke),
            jt.array(u),
            build_dof_indices(jt.array(node_index)),
        ).numpy()

        expected_flat = np.zeros((8 * 3, 2), dtype=np.float32)
        u_flat = u.reshape(8 * 3, 2)
        for element in node_index:
            dofs = (element[:, None] * 3 + np.arange(3)).reshape(-1)
            expected_flat[dofs] += ke @ u_flat[dofs]

        np.testing.assert_allclose(actual, expected_flat.reshape(8, 3, 2), rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
