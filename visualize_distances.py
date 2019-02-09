import argparse
import os

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
    titles = [f'{i}-{j + 1}' for i in range(10) for j in range(3)]

    for i, title, index in zip(range(len(indices)), titles, indices):
        batch = batches[index]
        plt.subplot(5, 6, (i + 1))
        plt.subplots_adjust(hspace=0.5)
        plt.imshow(np.reshape(batch, (28, 28)))
        plt.gray()
        plt.title(title, pad=2.0)
        plt.xticks(color=str(None))
        plt.yticks(color=str(None))
        plt.tick_params(length=0)
    head, ext = os.path.splitext(args.source)
    plt.savefig(f'{head}-choices.png')

    pairwise_distances = functions._pairwise_distances(embeddings[indices], dist_type=args.dist_type).data
    np.savetxt(f'{head}-distances.txt', pairwise_distances, fmt='%.2f')
    plt.close()

    plt.imshow(pairwise_distances)
    plt.xticks(np.arange(0, 30 + 1, 1), labels=titles, rotation=-90)
    plt.yticks(np.arange(0, 30 + 1, 1), labels=titles)
    plt.savefig(f'{head}-distances.png')


if __name__ == '__main__':
    main()
