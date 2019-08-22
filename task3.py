import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.backends import cuda
from chainer.training import extensions


class MLP(chainer.Chain):

    """多層パーセプトロン (MultiLayer Perceptron)"""

    def __init__(self):
        super().__init__()
        with self.init_scope():
            # 全結合層3つ
            self.fc1 = L.Linear(100)
            self.fc2 = L.Linear(100)
            self.fc3 = L.Linear(10)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc3(h)


class CNN(chainer.Chain):

    """畳み込みニューラルネットワーク (Convolutional Neural Network)"""

    def __init__(self):
        super().__init__()
        with self.init_scope():
            # 畳み込み層2つ
            self.conv1 = L.Convolution2D(None, out_channels=16, ksize=3, stride=1, pad=1)
            self.conv2 = L.Convolution2D(None, out_channels=32, ksize=3, stride=1, pad=1)
            # 最後の全結合層（0〜9まで10種類分類）
            self.fc = L.Linear(10)

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(h, 2)
        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(h, 2)
        return self.fc(h)


def train(model, optimizer, device, batchsize, epoch):
    # GPU利用準備
    if device >= 0:
        model.to_gpu(device)
        cuda.get_device_from_id(device).use()

    optimizer.setup(model)

    # MNIST データセットの読み込み
    train_data, test_data = chainer.datasets.get_mnist(ndim=3)
    train_iter = chainer.iterators.SerialIterator(train_data, batchsize)
    test_iter = chainer.iterators.SerialIterator(test_data, batchsize, repeat=False, shuffle=False)

    # trainerの初期化
    updater = training.updaters.StandardUpdater(train_iter, optimizer, device=device)
    trainer = training.Trainer(updater, (epoch, 'epoch'), out='results')

    # Test setでのevaluationの設定
    trainer.extend(extensions.Evaluator(test_iter, model, device=device))

    # 損失・精度等の出力の設定
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    # 学習を開始
    trainer.run()


model = L.Classifier(MLP())  # MLP() -> CNN() に変えてみてください
optimizer = chainer.optimizers.SGD()  # SGD() -> Adam() に変えてみてください
train(model, optimizer, device=0, batchsize=256, epoch=20)
