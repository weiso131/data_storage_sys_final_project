import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time


from GPUPowerLogger import GPUPowerLogger

# ----------------------------
# 超參數
# ----------------------------
batch_size = 64
epochs = 1
learning_rate = 0.01
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# MNIST 資料
# ----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# ----------------------------
# 模型
# ----------------------------

model_choice = input()

if model_choice == 'A':
    model = nn.Sequential(
        nn.Linear(784, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    ).to(device)
elif model_choice == 'B':
    model = nn.Sequential(
        nn.Linear(784, 500),
        nn.ReLU(),
        nn.Linear(500, 250),
        nn.ReLU(),
        nn.Linear(250, 10)
    ).to(device)

elif model_choice == 'C':
    model = nn.Sequential(
        nn.Linear(784, 1500),
        nn.ReLU(),
        nn.Linear(1500, 1000),
        nn.ReLU(),
        nn.Linear(1000, 500),
        nn.ReLU(),
        nn.Linear(500, 10),
    ).to(device)


optimizer = optim.SGD(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()



# ----------------------------
# 訓練 + 記錄 GPU 功耗
# ----------------------------
gpu_logger = GPUPowerLogger(gpu_index=0, interval=0.1)
gpu_logger.start()

elapsed_time = 0

records = []

for data, target in train_loader:
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()

    f_start = time.perf_counter()
    output = model(data.view(data.size(0), -1))
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    f_end = time.perf_counter()
    elapsed_time += f_end - f_start

    records.append((f_start, f_end))
        
gpu_logger.stop()
gpu_logger.join()
avg_power = gpu_logger.get_average_power(records)
energy = avg_power * elapsed_time # J = W * s

print(f"\n訓練總時間: {elapsed_time:.3f} s")
print(f"平均 GPU 功耗: {avg_power:.2f} W")
print(f"估算能耗: {energy:.2f} J")
