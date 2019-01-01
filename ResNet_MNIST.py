from functools import reduce
from keras import backend as K
from keras.layers import (Activation, Add, GlobalAveragePooling2D,
                          BatchNormalization, Conv2D, Dense, Flatten, Input,
                          MaxPooling2D)
from keras.models import Model
from keras.regularizers import l2
from functions import (basic_block, bottleneck_block, compose, ResNetConv2D, residual_blocks)
from keras.utils import plot_model
from mnist_preprocess import MNISTDataset
from keras.optimizers import SGD
import time
import numpy as np
import matplotlib.pyplot as plt
from build import ResnetBuilder
from cifar_10_preprocess import get_preprocessed_cifar10
from keras import losses

NB_CLASSES=10
NB_EPOCH = 20
BATCH_SIZE = 256
VERBOSE = 1

np.random.seed(None)

class ResNetMNIST:
    def __init__(self):
        self.MNIST_inputShape=(28, 28, 1)
        self.momentum = SGD(lr=0.1, decay=0.0001, momentum=0.9, nesterov=True)
        self.ResNetModel = ResnetBuilder.build_resnet_34(self.MNIST_inputShape, NB_CLASSES)
        self.ResNetModel.compile(optimizer=self.momentum, loss=losses.binary_crossentropy, metrics=['acc'])
    
    def train_mnist(self, output_graph=True):
        x_train, y_train, x_test, y_test = MNISTDataset().get_batch()
        
        trainDataAccuracy_array = []
        testDataAccuracy_array = []
        epoch_array = range(1, NB_EPOCH + 1)

        start_time = time.time()
        for epoch in range(NB_EPOCH):
            perm = np.random.permutation(x_train.shape[0])

            for i in range(0, x_train.shape[0], BATCH_SIZE):
                x_batch = x_train[perm[i : i + BATCH_SIZE]]
                y_batch = y_train[perm[i : i + BATCH_SIZE]]

                self.ResNetModel.train_on_batch(x_batch, y_batch)
            
            train_score = self.ResNetModel.evaluate(x_train, y_train, batch_size=BATCH_SIZE, verbose=VERBOSE)
            test_score = self.ResNetModel.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=VERBOSE)
            trainDataAccuracy_array.append(train_score[1])
            testDataAccuracy_array.append(test_score[1])
            interval = int(time.time() - start_time)
            print('epoch = {0:d} / {1:d} --- 実行時間 = {2:d}[sec] --- 1epochに掛かる平均時間 = {3:.2f}[sec]'.format(epoch + 1, NB_EPOCH, interval, interval / (epoch + 1)))
            print("Test score : {0:f} --- Test accuracy : {1:f}".format(test_score[0], test_score[1]))
        end_time = int(time.time() - start_time)

        if output_graph:
            plt.plot(epoch_array, trainDataAccuracy_array, label="train")
            plt.plot(epoch_array, testDataAccuracy_array, linestyle="--",label="test")
            plt.xlabel("epoch")
            plt.ylabel("accuracy")
            plt.title("ResNet with MNIST ({0:d}[sec])".format(end_time))
            plt.legend()
            plt.show()

def main():
    net = ResNetMNIST()
    net.train_mnist()

if __name__ == "__main__":
    main()