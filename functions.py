import chainer
import chainer.functions as F
import cupy as xp


def _pairwise_distances_l2(embeddings):
    """Compute the 2D matrix of distances between all the embeddings.

    Distance is defined by L2 norm: distance(x, y) := ||x - y||^2

    Args:
        embeddings: Variable with shape=(batch_size, embed_dim)

    Returns:
        pairwise_distances: Variable with shape=(batch_size, batch_size)
    """
    # Multiply 0.5 to embeddings because constrain distance in [0.0, 1.0].
    embeddings = 0.5 * embeddings

    # Get the dot product between all embeddings
    # shape (batch_size, batch_size)
    dot_product = F.matmul(embeddings, embeddings, transa=False, transb=True)

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    squared_norm = F.diagonal(dot_product)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = F.expand_dims(squared_norm, axis=0) - 2.0 * dot_product + F.expand_dims(squared_norm, axis=1)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = F.clip(distances, 0.0, 1.0)

    return distances


def _pairwise_distances_cos(embeddings):
    """Compute the 2D matrix of distances between all the embeddings.

    Distance is defined by cosine similarity: distance(x, y) := (1 - cos(theta)) / 2 = (1 - dot(x, y)) / 2

    Args:
        embeddings: Variable with shape=(batch_size, embed_dim)

    Returns:
        pairwise_distances: Variable with shape=(batch_size, batch_size)
    """
    # Get the dot product between all embeddings
    # shape (batch_size, batch_size)
    dot_product = F.matmul(embeddings, embeddings, transa=False, transb=True)

    # Compute the pairwise distance matrix
    distances = (1.0 - dot_product) * 0.5

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = F.clip(distances, 0.0, 1.0)

    return distances


def _pairwise_distances(embeddings, dist_type):
    """Compute the 2D matrix of distances between all the embeddings.

    If dist_type is 'l2', returns squared euclidean distance. distnace(a, b) := ||f(a) - f(b)||^2
    If dist_type is 'cos', returns distance computed by cosine similarity. distance(a, b) := (1 - cos(a, b)) / 2)

    Args:
        embeddings: Variable with shape=(batch_size, embed_dim)
        dist_type: definition of distance, 'l2' or 'cos'

    Returns:
        pairwise_distances: Variable with shape=(batch_size, batch_size)
    """
    return {
        'l2': _pairwise_distances_l2,
        'cos': _pairwise_distances_cos
    }[dist_type](embeddings)


def _get_anchor_positive_triplet_mask(labels):
    """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.

    Args:
        labels: xp.array with shape=(batch_size)

    Returns:
        mask: xp.array with shape=(batch_size, batch_size), dtype=xp.bool
    """
    # Check that i and j are distinct
    indices_not_equal = xp.logical_not(xp.diag(xp.ones(labels.size, dtype=xp.bool)))

    # Check if labels[i] == labels[j]
    # By using broadcasting:
    # Left side's shape (1, batch_size) => (*, batch_size)
    # Right side's shape (batch_size, 1) => (batch_size, *)
    labels_equal = xp.expand_dims(labels, axis=0) == xp.expand_dims(labels, axis=1)

    # Combine the two masks
    return indices_not_equal & labels_equal


def _get_anchor_negative_triplet_mask(labels):
    """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.

    Args:
        labels: xp.array with shape=(batch_size)

    Returns:
        mask: xp.array with shape=(batch_size, batch_size), dtype=xp.bool
    """
    # Check if labels[i] == labels[j]
    # By using broadcasting:
    # Left side's shape (1, batch_size) => (*, batch_size)
    # Right side's shape (batch_size, 1) => (batch_size, *)
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
    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    # By using broadcasting:
    # Left side's shape (batch_size, batch_size) => (batch_size, batch_size, *)
    # Right side's shape (batch_size, batch_size) => (batch_size, *, batch_size)
    positive_triplet_mask = _get_anchor_positive_triplet_mask(labels)
    negative_triplet_mask = _get_anchor_negative_triplet_mask(labels)

    # Combine the two masks
    # shape (batch_size, batch_size, batch_size)
    return xp.expand_dims(positive_triplet_mask, axis=2) * xp.expand_dims(negative_triplet_mask, axis=1)


def batch_all_triplet_loss(embeddings, labels, margin=0.2, dist_type='l2'):
    """Build the triplet loss over a batch of embeddings.

    We generate all the valid triplets and average the loss over the positive ones.

    Args:
        embeddings: Variable of shape=(batch_size, embed_dim)
        labels: labels of the batch, of size=(batch_size,)
        margin: margin for triplet loss
        dist_type: definition of distance, 'l2' or 'cos'

    Returns:
        triplet_loss: scalar Variable containing the triplet loss
    """
    # distance(f(xa), f(xp)) - distance(f(xa), f(xn)) + alpha
    pairwise_dist = _pairwise_distances(embeddings, dist_type)
    anchor_positive_dist = F.expand_dims(pairwise_dist, axis=2)
    anchor_negative_dist = F.expand_dims(pairwise_dist, axis=1)
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    # Set invalid triplet [i, j, k] to 0.
    mask = _get_triplet_mask(labels)
    triplet_loss = mask * triplet_loss

    # Ignore enough separated example pairs loss.
    triplet_loss = F.relu(triplet_loss)

    # Calculate mean of loss.
    total = F.sum(triplet_loss)
    count = xp.count_nonzero(triplet_loss.data)
    return total / count if (count > 0.0) else chainer.Variable(xp.array(0.0, dtype=xp.float32))


def validation_rate(embeddings, labels, threshold=0.2, dist_type='l2'):
    """ Calculate varidation rate metrics over a batch of embeddings.

    Args:
        embeddings: tensor with shape=(batch_size, embed_dim)
        labels: labels of the batch, with shape=(batch_size,)
        threshold: distance threshold decided two vector is close, scalar and positive
        dist_type: definition of distance, 'l2' or 'cos'

    Returns:
        triplet_loss: scalar Variable containing the triplet loss
    """
    mask = _get_anchor_positive_triplet_mask(labels)
    pairwise_distances = _pairwise_distances(embeddings, dist_type=dist_type).data
    numer = xp.sum((pairwise_distances < threshold) & mask)
    denom = xp.sum(mask)
    return numer / denom if (denom > 0.0) else 0.0


def false_accept_rate(embeddings, labels, threshold=0.2, dist_type='l2'):
    """ Calculate false accept rate over a batch of embeddings.

    Args:
        embeddings: tensor with shape=(batch_size, embed_dim)
        labels: labels of the batch, with shape=(batch_size,)
        threshold: distance threshold decided two vector is close, scalar and positive
        dist_type: definition of distance, 'l2' or 'cos'

    Returns:
        triplet_loss: scalar Variable containing the triplet loss
    """
    mask = _get_anchor_negative_triplet_mask(labels)
    pairwise_distances = _pairwise_distances(embeddings, dist_type=dist_type).data
    numer = xp.sum((pairwise_distances < threshold) & mask)
    denom = xp.sum(mask)
    return numer / denom if (denom > 0.0) else 0.0
