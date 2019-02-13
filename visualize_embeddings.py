import argparse
import os

import chainer
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def main():

    # Parse arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source', **{
        'type': str,
        'help': 'The embeddings.npy file path.',
        'required': True
    })
    parser.add_argument('-a', '--algorithm', **{
        'type': str,
        'help': 'Compression algorithm. "pca" or "tsne". default: pca.',
        'default': 'pca'
    })
    args = parser.parse_args()

    # Load embeddings file.
    embeddings = np.load(args.source)

    # Compress each embeddings to 2 dimension using PCA.
    if 2 < embeddings.shape[1]:
        compressor = {
            'pca': PCA,
            'tsne': TSNE
        }[args.algorithm]
        transforms = compressor(n_components=2).fit_transform(embeddings)
    else:
        transforms = embeddings

    # Load label data.
    _, val = chainer.datasets.get_mnist(ndim=3)
    labels = np.array([y for x, y in val])

    assert (len(labels) == len(transforms))

    for label in range(10):
        indices = np.where(labels == label)
        x, y = np.ravel(transforms[indices, 0]), np.ravel(transforms[indices, 1])
        plt.scatter(x, y, label=str(label))
    plt.legend()
    head, ext = os.path.splitext(args.source)
    plt.savefig(f'{head}-{args.algorithm}.png')


if __name__ == '__main__':
    main()
