import datetime
import os
import shutil
import yaml
from types import SimpleNamespace as Namespace

import chainer
import cupy as cp
from chainer.dataset.convert import concat_examples
from chainer.iterators import SerialIterator
from chainer.training import Trainer
from chainer.training.extensions import (Evaluator, LogReport, PlotReport,
                                         PrintReport, ProgressBar, dump_graph,
                                         snapshot_object)
from chainer.training.updaters import StandardUpdater

import functions
from model import FaceNet


class Classifier(chainer.Chain):
    def __init__(self, predictor):
        super(Classifier, self).__init__()

        with self.init_scope():
            self.predictor = predictor

    def forward(self, *args, **kwargs):
        batch, labels = args
        embeddings = self.predictor(batch)
        loss = functions.batch_all_triplet_loss(embeddings, labels)
        chainer.reporter.report({
            'loss': loss,
            'VAL': functions.validation_rate(embeddings, labels),
            'FAR': functions.false_accept_rate(embeddings, labels)
        }, self)
        return loss


def main():

    # Parse arguments.
    with open('params.yml') as stream:
        args = yaml.load(stream)

    # Prepare training data.
    train, val = chainer.datasets.get_mnist(ndim=3)

    # Prepare model.
    predictor = FaceNet()
    model = Classifier(predictor)
    if 0 <= args['gpu']:
        chainer.backends.cuda.get_device_from_id(args['gpu']).use()
        model.to_gpu()

    # Prepare optimizer.
    optimizer = chainer.optimizers.AdaDelta()
    optimizer.setup(model)

    # Training.
    timestamp = f'{datetime.datetime.now():%Y%m%d%H%M%S}'
    directory = f'./temp/{timestamp}/'
    os.makedirs(directory, exist_ok=True)
    shutil.copy('params.yml', f'{directory}params.yml')

    if args['memory'] == 'cpu' and 0 <= args['gpu']:
        def converter(batch, device=None, padding=None):
            return concat_examples([(cp.array(x), cp.array(y)) for x, y in batch], device=device, padding=padding)
    else:
        converter = concat_examples

    train_iter = SerialIterator(train, args['batch_size'])
    test_iter = SerialIterator(val, args['batch_size'], repeat=False, shuffle=False)
    updater = StandardUpdater(train_iter, optimizer, converter=converter)
    trainer = Trainer(updater, stop_trigger=(args['epochs'], 'epoch'), out=directory)
    trainer.extend(dump_graph('main/loss', out_name='model.dot'))
    trainer.extend(Evaluator(test_iter, model, converter=converter))
    trainer.extend(snapshot_object(target=model, filename='model-{.updater.epoch:04d}.npz'), trigger=(args['checkpoint_interval'], 'epoch'))
    trainer.extend(LogReport(log_name='log'))
    trainer.extend(PlotReport(y_keys=['main/loss', 'validation/main/loss'], x_key='epoch', file_name='loss.png'))
    trainer.extend(PlotReport(y_keys=['main/VAL', 'validation/main/VAL'], x_key='epoch', file_name='VAL.png'))
    trainer.extend(PlotReport(y_keys=['main/FAR', 'validation/main/FAR'], x_key='epoch', file_name='FAR.png'))
    trainer.extend(PrintReport(['epoch',
                                'main/loss', 'validation/main/loss',
                                'main/VAL', 'validation/main/VAL',
                                'main/FAR', 'validation/main/FAR',
                                'elapsed_time']))
    trainer.extend(ProgressBar(update_interval=1))
    trainer.run()


if __name__ == '__main__':
    main()
