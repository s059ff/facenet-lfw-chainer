import chainer
import chainer.functions as F
import chainer.links as L


class FaceNet(chainer.Chain):

    def __init__(self):
        super(FaceNet, self).__init__()

        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 64, ksize=(3, 3), pad=1)
            self.bn1 = L.BatchNormalization(64)
            self.conv2 = L.Convolution2D(None, 128, ksize=(3, 3), pad=1)
            self.bn2 = L.BatchNormalization(128)
            self.conv3 = L.Convolution2D(None, 256, ksize=(3, 3), pad=1)
            self.bn3 = L.BatchNormalization(256)
            self.conv4 = L.Convolution2D(None, 512, ksize=(3, 3), pad=1)
            self.bn4 = L.BatchNormalization(512)
            self.fc5 = L.Linear(None, 128)

    def __call__(self, h):
        h = F.max_pooling_2d(F.relu(self.bn1(self.conv1(h))), (2, 2))
        h = F.max_pooling_2d(F.relu(self.bn2(self.conv2(h))), (2, 2))
        h = F.max_pooling_2d(F.relu(self.bn3(self.conv3(h))), (2, 2))
        h = F.max_pooling_2d(F.relu(self.bn4(self.conv4(h))), (2, 2))
        h = self.fc5(h)
        h = F.normalize(h)
        return h
