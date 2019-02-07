import cupy as xp
import chainer.functions as F


def _pairwise_distances(embeddings):
    dot = F.matmul(embeddings, embeddings, transa=False, transb=True)
    squared_norm = F.diagonal(dot)
    distances = F.expand_dims(squared_norm, axis=0) - 2.0 * dot + F.expand_dims(squared_norm, axis=1)
    return distances


def _get_anchor_positive_triplet_mask(labels):
    return xp.where(xp.expand_dims(labels, axis=0) == xp.expand_dims(labels, axis=1), 1.0, 0.0)


def _get_anchor_negative_triplet_mask(labels):
    return xp.where(xp.expand_dims(labels, axis=0) != xp.expand_dims(labels, axis=1), 1.0, 0.0)


def _get_triplet_mask(labels):
    conditions = xp.expand_dims(labels, axis=0) == xp.expand_dims(labels, axis=1)
    return xp.expand_dims(xp.where(conditions, 1.0, 0.0), axis=2) * xp.expand_dims(xp.where(conditions, 0.0, 1.0), axis=1)


def batch_all_triplet_loss(embeddings, labels, margin=0.2):
    pairwise_dist = _pairwise_distances(embeddings)
    anchor_positive_dist = F.expand_dims(pairwise_dist, axis=2)
    anchor_negative_dist = F.expand_dims(pairwise_dist, axis=1)
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    mask = _get_triplet_mask(labels)
    triplet_loss = mask * triplet_loss

    triplet_loss = F.relu(triplet_loss)
    num_positive_triplets = xp.count_nonzero(triplet_loss.data) + 1e-9

    return F.sum(triplet_loss) / num_positive_triplets


def validation_rate(embeddings, labels, threshold=0.2):
    mask = _get_anchor_positive_triplet_mask(labels)
    distances = _pairwise_distances(embeddings).data
    return xp.sum((distances < threshold) & (mask == 1)), xp.sum(mask)


def false_accept_rate(embeddings, labels, threshold=0.2):
    mask = _get_anchor_negative_triplet_mask(labels)
    distances = _pairwise_distances(embeddings).data
    return xp.sum((distances < threshold) & (mask == 1)), xp.sum(mask)


if __name__ == "__main__":

    # Check if mask of triplet loss is correctly.
    labels = xp.array([0, 1, 1, 0], xp.int)
    mask = _get_triplet_mask(labels)
    expected = xp.zeros_like(mask, dtype=xp.float)
    for i in range(len(labels)):
        for j in range(len(labels)):
            for k in range(len(labels)):
                if labels[i] == labels[j] and labels[i] != labels[k]:
                    expected[i, j, k] = 1.
    assert (mask == expected).all()

    # Check if triplet loss decrease.
    # Check if VAL metrics increase.
    # Check if FAR metrics decrease.
    embeddings = xp.array([[0.0, 0.0], [1.0, 1.0], [0.2, 0.2], [0.4, 0.4]], xp.float)
    loss1 = batch_all_triplet_loss(embeddings, labels, 0.0)
    numer, denom = validation_rate(embeddings, labels)
    val1 = numer / denom
    numer, denom = false_accept_rate(embeddings, labels)
    far1 = numer / denom

    embeddings = xp.array([[0.0, 0.0], [1.0, 1.0], [0.4, 0.4], [0.1, 0.1]], xp.float)
    loss2 = batch_all_triplet_loss(embeddings, labels, 0.0)
    numer, denom = validation_rate(embeddings, labels)
    val2 = numer / denom
    numer, denom = false_accept_rate(embeddings, labels)
    far2 = numer / denom

    embeddings = xp.array([[0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [0.0, 0.0]], xp.float)
    loss3 = batch_all_triplet_loss(embeddings, labels, 0.0)
    numer, denom = validation_rate(embeddings, labels)
    val3 = numer / denom
    numer, denom = false_accept_rate(embeddings, labels)
    far3 = numer / denom

    print(loss1, loss2, loss3)
    print(val1, val2, val3)
    print(far1, far2, far3)

    assert loss1.data >= loss2.data and loss2.data >= loss3.data
    assert val1 <= val2 and val2 <= val3
    assert far1 >= far2 and far2 >= far3
