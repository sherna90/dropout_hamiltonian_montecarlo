import chainer
import chainer.functions as F
import chainer.links as L

from chainer.datasets import cifar
from chainer import iterators
from chainer import optimizers
from chainer import training
from chainer.training import extensions


class MyModel(chainer.Chain):
    def __init__(self,n_in, n_out):
        super(MyModel, self).__init__()
        with self.init_scope():
            self.conv1=L.Convolution2D(n_in, 32, 3, 3, 1)
            self.conv2=L.Convolution2D(32, 64, 3, 3, 1)
            self.conv3=L.Convolution2D(64, 64, 3, 3, 1)
            self.conv4=L.Convolution2D(64, 512, 3, 3, 1)
            self.fc5=L.Linear(512, 512)
            self.fc6=L.Linear(512, n_out)

    def __call__(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(h, 2, 2)
        h = F.dropout(h, ratio=0.25)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(h, 2, 2)
        h = F.dropout(h, ratio=0.25)
        h = self.fc5(h)
        h = F.dropout(h, ratio=0.5)
        h = self.fc6(h)
        return h


def train(model_object, batchsize=64, gpu_id=0, max_epoch=100):
    train, test = cifar.get_cifar10()
    train_iter = iterators.SerialIterator(train, batchsize)
    test_iter = iterators.SerialIterator(test, batchsize, False, False)
    model = L.Classifier(model_object)
    if gpu_id >=0:
        model.to_gpu(gpu_id)
    optimizer = optimizers.Adam()
    optimizer.setup(model)
    updater = training.StandardUpdater(train_iter, optimizer, device=gpu_id)
    trainer = training.Trainer(updater, (max_epoch, 'epoch'), out='{}_cifar10_result'.format(model_object.__class__.__name__))

    class TestModeEvaluator(extensions.Evaluator):
        def evaluate(self):
            model = self.get_target('main')
            ret = super(TestModeEvaluator, self).evaluate()
            return ret

    trainer.extend(extensions.LogReport())
    trainer.extend(TestModeEvaluator(test_iter, model, device=gpu_id))
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy', 'validation/main/loss', 'validation/main/accuracy', 'elapsed_time']))
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], x_key='epoch', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], x_key='epoch', file_name='accuracy.png'))
    trainer.run()
    del trainer
    return model

gpu_id = 0  # Set to -1 if you don't have a GPU
#model = train(MyModel(3072,10), gpu_id=gpu_id)