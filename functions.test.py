import unittest

import cupy as xp

import functions


class TestFunctions(unittest.TestCase):

    def test_get_triplet_mask(self):
        labels = xp.array([0, 1, 1, 0], xp.int)
        mask = functions._get_triplet_mask(labels)
        expected = xp.zeros_like(mask, dtype=xp.bool)
        for i in range(len(labels)):
            for j in range(len(labels)):
                for k in range(len(labels)):
                    if i != j and labels[i] == labels[j] and labels[i] != labels[k]:
                        expected[i, j, k] = True
        self.assertSequenceEqual(mask.tolist(), expected.tolist())

    def test_triplet_loss_l2(self):
        self._test_triplet_loss(dist_type='l2')

    def test_triplet_loss_cos(self):
        self._test_triplet_loss(dist_type='cos')

    def _test_triplet_loss(self, dist_type):
        labels = xp.array([0, 1, 1, 0], xp.int)

        embeddings1 = xp.array([[0.0, 0.0], [1.0, 1.0], [0.2, 0.2], [0.4, 0.4]], xp.float32)
        loss1 = functions.batch_all_triplet_loss(embeddings1, labels, 0.0, dist_type=dist_type)
        val1 = functions.validation_rate(embeddings1, labels, dist_type=dist_type)
        far1 = functions.false_accept_rate(embeddings1, labels, dist_type=dist_type)

        embeddings2 = xp.array([[0.0, 0.0], [1.0, 1.0], [0.4, 0.4], [0.1, 0.1]], xp.float32)
        loss2 = functions.batch_all_triplet_loss(embeddings2, labels, 0.0, dist_type=dist_type)
        val2 = functions.validation_rate(embeddings2, labels, dist_type=dist_type)
        far2 = functions.false_accept_rate(embeddings2, labels, dist_type=dist_type)

        embeddings3 = xp.array([[0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [0.0, 0.0]], xp.float32)
        loss3 = functions.batch_all_triplet_loss(embeddings3, labels, 0.0, dist_type=dist_type)
        val3 = functions.validation_rate(embeddings3, labels, dist_type=dist_type)
        far3 = functions.false_accept_rate(embeddings3, labels, dist_type=dist_type)

        self.assertGreaterEqual(loss1.data, loss2.data)
        self.assertGreaterEqual(loss2.data, loss3.data)
        print(loss1.data, loss2.data, loss3.data)

        self.assertLessEqual(val1, val2)
        self.assertLessEqual(val2, val3)
        print(val1, val2, val3)

        self.assertGreaterEqual(far1, far2)
        self.assertGreaterEqual(far2, far3)
        print(far1, far2, far3)


if __name__ == "__main__":
    unittest.main()
