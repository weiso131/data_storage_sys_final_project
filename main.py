import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision import transforms
import torch.nn.functional as F

from pipelayer import *

root = "data"

USE_PIPELAYER = True

mnist_train = MNIST(
    root=root,
    train=True,
    download=True,
    transform=transforms.ToTensor()  # 轉成 tensor
)

mnist_test = MNIST(
    root=root,
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

dtype = torch.float32

if USE_PIPELAYER:
    dtype = torch.float16

X_train = (mnist_train.data / 255.0).to(dtype)
y_train = mnist_train.targets
y_train_onehot = torch.zeros(y_train.shape[0], 10, dtype=dtype)
y_train_onehot[torch.arange(y_train.shape[0]), y_train] = 1.0

batch_size = 64
lr = 0.001

pipelayer = PipeLayer(device=torch.device("cuda"))
pipelayer.push_layer(in_features=784, out_features=1000, dtype=dtype)
pipelayer.push_layer(in_features=1000, out_features=10, dtype=dtype)

X_train = X_train.view(-1, 784)
X_batches = torch.split(X_train, batch_size)
y_batches = torch.split(y_train_onehot, batch_size)

print(X_train.numel())

pipelayer.pipeline_train(X_batches, y_batches, 5)
pipelayer.without_pipeline_train(X_batches, y_batches, 5)
pipelayer.pipeline_test(X_batches, y_batches)
pipelayer.without_pipeline_test(X_batches, y_batches)
