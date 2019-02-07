import chainer
import chainer.functions as F
import cupy as xp


def _pairwise_distances(embeddings):
    """Compute the 2D matrix of distances between all the embeddings.

    Args:
        embeddings: Variable with shape=(batch_size, embed_dim)

    Returns:
        pairwise_distances: Variable with shape=(batch_size, batch_size)
    """
    dot = F.matmul(embeddings, embeddings, transa=False, transb=True)
    squared_norm = F.diagonal(dot)
    pairwise_distances = F.expand_dims(squared_norm, axis=0) - 2.0 * dot + F.expand_dims(squared_norm, axis=1)
    return pairwise_distances


def _get_anchor_positive_triplet_mask(labels):
    """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.

    Args:
        labels: xp.array with shape=(batch_size)

    Returns:
        mask: xp.array with shape=(batch_size, batch_size), dtype=xp.bool
    """
    return (xp.expand_dims(labels, axis=0) == xp.expand_dims(labels, axis=1)) & xp.logical_not(xp.diag(xp.ones(labels.size, dtype=xp.bool)))


def _get_anchor_negative_triplet_mask(labels):
    """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.

    Args:
        labels: xp.array with shape=(batch_size)

    Returns:
        mask: xp.array with shape=(batch_size, batch_size), dtype=xp.bool
    """
    return xp.expand_dims(labels, axis=0) != xp.expand_dims(labels, axis=1)


def _get_triplet_mask(labels):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.

    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]

    Args:
        labels: xp.array with shape=(batch_size)

    Returns:
        mask: xp.array with shape=(batch_size, batch_size, batch_size), dtype=xp.bool
    """
    mask1 = _get_anchor_positive_triplet_mask(labels)
    mask2 = _get_anchor_negative_triplet_mask(labels)
    return xp.expand_dims(mask1, axis=2) * xp.expand_dims(mask2, axis=1)


def batch_all_triplet_loss(embeddings, labels, margin=0.2):
    """Build the triplet loss over a batch of embeddings.

    We generate all the valid triplets and average the loss over the positive ones.

    Args:
        embeddings: Variable of shape=(batch_size, embed_dim)
        labels: labels of the batch, of size=(batch_size,)
        margin: margin for triplet loss

    Returns:
        triplet_loss: scalar Variable containing the triplet loss
    """
    pairwise_dist = _pairwise_distances(embeddings)
    anchor_positive_dist = F.expand_dims(pairwise_dist, axis=2)
    anchor_negative_dist = F.expand_dims(pairwise_dist, axis=1)
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    mask = _get_triplet_mask(labels)
    triplet_loss = mask * triplet_loss

    triplet_loss = F.relu(triplet_loss)

    total = F.sum(triplet_loss)
    count = xp.count_nonzero(triplet_loss.data)
    return total / count if (count > 0.0) else chainer.Variable(xp.array(0.0, dtype=xp.float32))


def validation_rate(embeddings, labels, threshold=0.2):
    """ Calculate varidation rate metrics over a batch of embeddings.

    Args:
        embeddings: tensor with shape=(batch_size, embed_dim)
        labels: labels of the batch, with shape=(batch_size,)
        threshold: distance threshold decided two vector is close, scalar and positive

    Returns:
        triplet_loss: scalar Variable containing the triplet loss
    """
    mask = _get_anchor_positive_triplet_mask(labels)
    pairwise_distances = _pairwise_distances(embeddings).data
    numer = xp.sum((pairwise_distances < threshold) & mask)
    denom = xp.sum(mask)
    return numer / denom if (denom > 0.0) else 0.0


def false_accept_rate(embeddings, labels, threshold=0.2):
    """ Calculate false accept rate over a batch of embeddings.

    Args:
        embeddings: tensor with shape=(batch_size, embed_dim)
        labels: labels of the batch, with shape=(batch_size,)
        threshold: distance threshold decided two vector is close, scalar and positive

    Returns:
        triplet_loss: scalar Variable containing the triplet loss
    """
    mask = _get_anchor_negative_triplet_mask(labels)
    pairwise_distances = _pairwise_distances(embeddings).data
    numer = xp.sum((pairwise_distances < threshold) & mask)
    denom = xp.sum(mask)
    return numer / denom if (denom > 0.0) else 0.0
