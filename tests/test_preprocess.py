import unittest

import numpy as np

from datagen.generate import process_one_voxel
from train.dataset import distributed_indices, sparse_collate


class PreprocessTest(unittest.TestCase):
    def test_distributed_indices_are_disjoint_and_balanced(self):
        shards = [
            distributed_indices(20, 2, True, True, rank, 3, seed=7)
            for rank in range(3)
        ]

        self.assertEqual([len(shard) for shard in shards], [6, 6, 6])
        self.assertEqual(len(set(np.concatenate(shards).tolist())), 18)
        self.assertTrue(set(shards[0]).isdisjoint(shards[1]))
        self.assertTrue(set(shards[0]).isdisjoint(shards[2]))
        self.assertTrue(set(shards[1]).isdisjoint(shards[2]))

    def test_periodic_connectivity_and_batch_offsets(self):
        voxel = np.zeros((4, 4, 4), dtype=bool)
        voxel[0, 0, 0] = True
        voxel[1, 0, 0] = True
        sample = process_one_voxel(voxel, 4)
        coords, node_type, node_index, _ = sample

        self.assertEqual(coords.shape, (12, 3))
        self.assertEqual(node_type.shape, (12, 8))
        self.assertEqual(node_index.shape, (2, 8))

        batch = sparse_collate([sample, sample])
        self.assertEqual(batch["coord"].shape, (24, 4))
        self.assertEqual(batch["node_index"].shape, (4, 8))
        self.assertLess(batch["node_index"][:2].max(), 12)
        self.assertGreaterEqual(batch["node_index"][2:].min(), 12)


if __name__ == "__main__":
    unittest.main()
