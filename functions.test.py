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

    def test_triplet_loss(self):
        labels = xp.array([0, 1, 1, 0], xp.int)

        embeddings1 = xp.array([[0.0, 0.0], [1.0, 1.0], [0.2, 0.2], [0.4, 0.4]], xp.float32)
        loss1 = functions.batch_all_triplet_loss(embeddings1, labels, 0.0)
        val1 = functions.validation_rate(embeddings1, labels)
        far1 = functions.false_accept_rate(embeddings1, labels)

        embeddings2 = xp.array([[0.0, 0.0], [1.0, 1.0], [0.4, 0.4], [0.1, 0.1]], xp.float32)
        loss2 = functions.batch_all_triplet_loss(embeddings2, labels, 0.0)
        val2 = functions.validation_rate(embeddings2, labels)
        far2 = functions.false_accept_rate(embeddings2, labels)

        embeddings3 = xp.array([[0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [0.0, 0.0]], xp.float32)
        loss3 = functions.batch_all_triplet_loss(embeddings3, labels, 0.0)
        val3 = functions.validation_rate(embeddings3, labels)
        far3 = functions.false_accept_rate(embeddings3, labels)

        self.assertGreaterEqual(loss1.data, loss2.data)
        self.assertGreaterEqual(loss2.data, loss3.data)

        self.assertLessEqual(val1, val2)
        self.assertLessEqual(val2, val3)

        self.assertGreaterEqual(far1, far2)
        self.assertGreaterEqual(far2, far3)


if __name__ == "__main__":
    unittest.main()
