import argparse
import os
import itertools

import chainer
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

import functions


def main():

    # Parse arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source', **{
        'type': str,
        'help': 'The embeddings.npy file path.',
        'required': True
    })
    parser.add_argument('-d', '--dist_type', **{
        'type': str,
        'required': True
    })
    args = parser.parse_args()

    # Load embeddings file.
    embeddings = np.load(args.source)

    # Compress each embeddings to 2 dimension using PCA.
    if 2 < embeddings.shape[1]:
        pca = PCA(n_components=2)
        pca.fit(embeddings)

    # Load inputs data.
    _, val = chainer.datasets.get_mnist(ndim=3)
    batches = np.array([x for x, y in val])
    labels = np.array([y for x, y in val])

    np.random.seed(0x12345)
    indices = np.ravel([np.random.choice(np.ravel(np.argwhere(labels == i)), size=3) for i in range(10)])

    head, ext = os.path.splitext(args.source)

    pairwise_distances = functions._pairwise_distances(embeddings[indices], dist_type=args.dist_type).data
    np.savetxt(f'{head}-distances.txt', pairwise_distances, fmt='%.2f')

    fig, axes = plt.subplots(31, 31, figsize=(20, 20))
    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    plt.gray()

    ax = axes[0, 0]
    ax.set_xticks([]), ax.set_yticks([])

    for i, index in zip(range(30), indices):
        batch = batches[index]
        ax = axes[i + 1, 0]
        ax.set_xticks([]), ax.set_yticks([])
        ax.imshow(batch.reshape((28, 28)), vmin=0.0, vmax=1.0)
        ax = axes[0, i + 1]
        ax.set_xticks([]), ax.set_yticks([])
        ax.imshow(batch.reshape((28, 28)), vmin=0.0, vmax=1.0)

    for i, j in itertools.product(range(30), range(30)):
        value = np.array([[pairwise_distances[i, j]]])
        ax = axes[i + 1, j + 1]
        ax.set_xticks([]), ax.set_yticks([])
        ax.imshow(value, vmin=0.0, vmax=1.0)
    plt.savefig(f'{head}-distances.png')
    plt.close()


if __name__ == '__main__':
    main()
