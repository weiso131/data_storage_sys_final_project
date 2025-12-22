from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

NULL = 0

class OperationReturn:
    def __init__(self, output, reram_time: float, reram_energy: float, arithmetic: int):
        self.output = output
        self.reram_time = reram_time
        self.reram_energy = reram_energy
        self.arithmetic = arithmetic


class LieanrLayer:
    def __init__(self, in_features: int, out_features: int, dtype=torch.float32, device=torch.device("cpu")):
        self.linear = nn.Linear(in_features=in_features, out_features=out_features, bias=True).to(dtype).to(device)
        self.linear.weight.requires_grad_(False)
        self.linear.bias.requires_grad_(False)
        self.dw = torch.zeros_like(self.linear.weight, dtype=dtype).to(device)
        self.db = torch.zeros_like(self.linear.bias, dtype=dtype).to(device)
    
    def update(self, lr: float, batch_size: int):
        self.linear.weight -= self.dw * lr / batch_size
        self.linear.bias -= self.db * lr / batch_size
        self.dw = torch.zeros_like(self.linear.weight, dtype=self.linear.weight.dtype)
        self.db = torch.zeros_like(self.linear.bias, dtype=self.linear.bias.dtype)

    def forward(self, x, func=F.relu, reram_read_time=29.31, reram_write_time=50.88, 
                        reram_read_energy=1.08, reram_write_energy=3910):
        d = func(self.linear(x))

        time = 16 * ((reram_read_time + reram_write_time) * x.numel() + \
                        reram_write_time * d.numel())
        energy = 16 * ((reram_read_energy + reram_write_energy) * x.numel() + \
                       reram_write_energy * d.numel())
        # read from memory subarrays, write to morphable subarrays, write to memory subarrays
        arithmetic = 0
        return OperationReturn(d, time, energy, arithmetic)
    
    def backward(self, last_d, loss, cal_last_loss=True, reram_read_time=29.31, reram_write_time=50.88, 
                        reram_read_energy=1.08, reram_write_energy=3910):
        self.dw += loss.view(-1, 1) @ last_d.view(1, -1)
        self.db += loss
        time = 16 * ((reram_read_time + reram_write_time) * last_d.numel() + \
                    (reram_read_time + reram_write_time) * loss.numel())
        energy = 16 * ((reram_read_energy + reram_write_energy) * last_d.numel() + \
                    (reram_read_energy + reram_write_energy) * loss.numel())
        arithmetic = 0

        if cal_last_loss:
            last_loss = (self.linear.weight.T @ loss) * (last_d > 0).to(last_d.dtype)
            time += 16 * reram_write_time * last_loss.numel()
            energy += 16 * ((reram_read_energy + reram_write_energy) * last_d.numel() + \
                    (reram_read_energy + reram_write_energy) * loss.numel() + \
                    reram_write_energy * last_loss.numel())
            return OperationReturn(last_loss, time, energy, arithmetic)

        else:
            return OperationReturn(0, time, energy, arithmetic)



class PipeLayer:
    def __init__(self, reram_read_time=29.31, reram_write_time=50.88, 
                        reram_read_energy=1.08, reram_write_energy=3910, device=torch.device("cpu")):
        self.layers = []
        self.reram_read_time = reram_read_time
        self.reram_write_time = reram_write_time
        self.reram_read_energy = reram_read_energy
        self.reram_write_energy = reram_write_energy
        self.device = device
    def push_layer(self, in_features: int, out_features: int, dtype=torch.float32):
        layer =LieanrLayer(in_features, out_features, dtype, self.device)
        self.layers.append(layer)

    def without_pipeline_train(self, x_batch, y_batch, epoch: int, lr=0.001):
        time = 0
        energy = 0
        arithmetic = 0
        for _ in range(epoch):
            acc = 0
            cnt = 0
            for batch in range(len(x_batch)):
                x = x_batch[batch].to(self.device)
                y = y_batch[batch].to(self.device)
                batch_size = x.shape[0]

                for i in range(batch_size):
                    d_list = [x[i]]
                    for l in range(len(self.layers)):
                        func = F.relu
                        if (l == len(self.layers) - 1):
                            func = F.softmax

                        output = self.layers[l].forward(d_list[-1], func)
                        d_list.append(output.output)
                        time += output.reram_time
                        energy += output.reram_energy
                        arithmetic += output.arithmetic
                    
                    cnt += 1
                    if torch.argmax(d_list[-1]) == torch.argmax(y[i]):
                        acc += 1
                    loss = d_list[-1] - y[i]

                    time += 16 * ((self.reram_read_time + self.reram_write_time) * (d_list[-1].numel() + y[i].numel()) + \
                                   self.reram_write_time * loss.numel())
                    energy += 16 * ((self.reram_read_energy + self.reram_write_energy) * (d_list[-1].numel() + y[i].numel()) + \
                                    self.reram_write_energy * loss.numel())

                    d_list.pop()
                        
                    for l in range(len(self.layers) - 1, -1, -1):
                        output = self.layers[l].backward(d_list.pop(), loss, l != 0)
                        loss = output.output
                        time += output.reram_time
                        energy += output.reram_energy
                        arithmetic += output.arithmetic
                
                for l in range(len(self.layers)):
                    time += 16 * self.reram_write_time * \
                        (self.layers[l].linear.weight.numel() + self.layers[l].linear.bias.numel())
                    energy += 16 * self.reram_write_energy * \
                        (self.layers[l].linear.weight.numel() + self.layers[l].linear.bias.numel())
                    self.layers[l].update(lr, batch_size)
            print(f"acc: {acc / cnt * 100}%")
        print(f"time: {time / 1000000} ms, energy: {energy / 1000000000} mj")
    def without_pipeline_test(self, x_batch, y_batch):
        time = 0
        energy = 0
        arithmetic = 0
        acc = 0
        cnt = 0
        for batch in range(len(x_batch)):
            x = x_batch[batch].to(self.device)
            y = y_batch[batch].to(self.device)
            batch_size = x.shape[0]

            for i in range(batch_size):
                d_list = [x[i]]
                for l in range(len(self.layers)):
                    func = F.relu
                    if (l == len(self.layers) - 1):
                        func = F.softmax

                    output = self.layers[l].forward(d_list[-1], func)
                    d_list.append(output.output)
                    time += output.reram_time
                    energy += output.reram_energy
                    arithmetic += output.arithmetic
                
                cnt += 1
                if torch.argmax(d_list[-1]) == torch.argmax(y[i]):
                    acc += 1
            
        print(f"test:\nacc: {acc / cnt * 100}%")
        print(f"time: {time / 1000000} ms, energy: {energy / 1000000000} mj")
    def pipeline_test(self, x_batch, y_batch):
        acc = 0
        cnt = 0
        time = 0
        energy = 0
        for batch in range(len(x_batch)):
            x = x_batch[batch].to(self.device)
            y = y_batch[batch].to(self.device)
            batch_size = x.shape[0]
            
            data_cnt = 1
            input_queue = [NULL] * (len(self.layers) + 1)
            input_queue[0] = (0, x[0])
            
            swap_queue = [NULL] * (len(self.layers) + 1)

            while True:
                input_cnt = 0
                if data_cnt < batch_size:
                    swap_queue[0] = (data_cnt, x[data_cnt])
                    data_cnt += 1
                    input_cnt += 1
                predict = NULL
                parallel_time = 0
                for l in range(len(self.layers)):
                    if (input_queue[l] == NULL):
                        continue                    
                    if (l != len(self.layers) - 1):
                        idx, d = input_queue[l]                
                        output = self.layers[l].forward(d, F.relu)
                        parallel_time = max(parallel_time, output.reram_time)
                        energy += output.reram_energy
                        swap_queue[l + 1] = (idx, output.output)
                        input_cnt += 1
                    else:
                        idx, d = input_queue[l]                
                        output = self.layers[l].forward(d, F.softmax)
                        predict = (idx, output.output)
                        energy += output.reram_energy
                        parallel_time = max(parallel_time, output.reram_time)
                time += parallel_time
                input_queue = swap_queue
                swap_queue = [NULL] * (len(self.layers) + 1)
                if predict != NULL:
                    idx, result = predict
                    cnt += 1
                    if torch.argmax(result) == torch.argmax(y[idx]):
                        acc += 1
                if input_cnt == 0:
                    break
        print(f"pipeline test:\nacc: {acc / cnt * 100}%")
        print(f"time:{time / 1000000} ms, energy:{energy / 1000000000} mj")
