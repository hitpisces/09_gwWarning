# -*- coding: utf-8 -*-
# @Time    : 19/12/9 16:38
# @Author  : Jay Lam
# @File    : pytorchTest.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1).cuda()
y = x.pow(2) + 0.3 * torch.rand(x.size()).cuda()

net1 = torch.nn.Sequential(
    torch.nn.Linear(1, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 1)
)
optimizer = torch.optim.SGD(net1.parameters(), lr=0.5)
lossFunc = torch.nn.MSELoss()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("当前使用的GPU是：" + str(torch.cuda.get_device_name(torch.cuda.current_device())))

net1.to(device)

for t in range(1000):
    prediction = net1(x)
    loss = lossFunc(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(loss.data.cpu().numpy())

plt.scatter(x.data.cpu().numpy(), y.data.cpu().numpy())
plt.show()
