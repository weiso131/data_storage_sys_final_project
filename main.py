import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision import transforms
import torch.nn.functional as F

root = "data"

USE_PIPELAYER = False

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

linear1 = nn.Linear(in_features=784, out_features=1000, bias=True).to(dtype)
linear2 = nn.Linear(in_features=1000, out_features=10, bias=True).to(dtype)

dw1 = torch.zeros_like(linear1.weight, dtype=dtype)
db1 = torch.zeros_like(linear1.bias, dtype=dtype)
dw2 = torch.zeros_like(linear2.weight, dtype=dtype)
db2 = torch.zeros_like(linear2.bias, dtype=dtype)

batch_size = 64

lr = 0.001
with torch.no_grad():
    for t in range(5):
        acc = 0
        for i in range(X_train.shape[0]):     
            d0 = X_train[i].view(-1)
            d1 = F.relu(linear1(d0))
            
            d2 = F.softmax(linear2(d1))

            if torch.argmax(d2) == y_train[i]:
                acc += 1

            loss2 = d2 - y_train_onehot[i] # CrossEntropy
            # A1
            dw2 += loss2.view(-1, 1) @ d1.view(1, -1)
            db2 += loss2

            # A2
            loss1 = (linear2.weight.T @ loss2) * (d1 > 0).to(d1.dtype) # relu'(x) == relu1(relu(x))

            # A1
            dw1 += loss1.view(-1, 1) @ d0.view(1, -1)
            db1 += loss1

            if (i != 0 and i % batch_size == 0) or (i + 1) == X_train.shape[0]:
                linear1.weight -= dw1 * lr / batch_size
                linear1.bias -= db1 * lr / batch_size
                linear2.weight -= dw2 * lr / batch_size
                linear2.bias -= db2 * lr / batch_size

                dw1 = torch.zeros_like(linear1.weight, dtype=dtype)
                db1 = torch.zeros_like(linear1.bias, dtype=dtype)
                dw2 = torch.zeros_like(linear2.weight, dtype=dtype)
                db2 = torch.zeros_like(linear2.bias, dtype=dtype)
        print(f"acc: {acc / X_train.shape[0] * 100}%")
