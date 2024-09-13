import numpy as np
import tensorflow as tf
import os
import sys
import copy
import torch
import torchvision
from tensorflow import float32
import torch.nn as nn
from torch.autograd import Variable
from matplotlib import pyplot as plt
from PIL import Image
from torchvision.transforms import ToPILImage

# -------------------定义L2攻击----------------------
class L2_attack:
    def __init__(self, model, c=1, confidence=0, steps=50, lr=0.01):
        self.c = c  # 常数c
        self.confidence = confidence  # 对抗样本的置信度
        self.steps = steps  # 步数
        self.lr = lr  # 学习率

    def forward(self, image, label):
        image = image.clone().detach().to(image.device)
        label = label.clone().detach().to(label.device)

        #  计算w，w=arctan(2*image-1)
        w = self.inverse_tanh_space(image).detach()
        w.requires_grad = True

        #  保存最好的攻击图像与最小的L2距离
        best_attack = image.clone().detach()
        best_L2 = 1e10 * torch.ones((len(image))).to(image.device)

        old_loss = 1e10
        dim = len(image.shape)

        MSELoss = nn.MSELoss(reduction="none")
        Flatten = nn.Flatten()

        #  对w进行优化
        optimizer = torch.optim.Adam([w], lr=self.lr)

        for step in range(self.steps):
            # 生成对抗图像
            attack_image = self.tanh_space(w)  # x+σ=(tanh(w)+1)/2

            # 计算对抗样本与原样本的L2距离
            cur_L2 = MSELoss(Flatten(attack_image), Flatten(image)).sum(dim=1)
            L2_loss = cur_L2.sum()

            output = model(attack_image)
            f_loss = self.f(output, label).sum()

            #  目标函数=L2距离+c*f
            new_loss = L2_loss + self.c * f_loss

            optimizer.zero_grad()
            new_loss.backward()
            optimizer.step()

            #  更新对抗图像
            prob = torch.argmax(output.detach(), 1)
            suc = (prob != label).float()  # 无目标攻击，只需要对抗图像的类别与原图像类别不同即可

            #  过滤攻击不成功或者L2_loss没有减小的图像
            tmp = suc * (best_L2 > cur_L2.detach())
            best_L2 = tmp * cur_L2.detach() + (1 - tmp) * best_L2

            #  print(tmp) :  tensor([0.], device='cuda:0')
            tmp = tmp.view([-1] + [1] * (dim - 1))
            #  print([-1] + [1] * (dim - 1))
            #  print(tmp) :  tensor([[[[0.]]]], device='cuda:0')
            best_attack = tmp * attack_image.detach() + (1 - tmp) * best_attack

            #  如果loss不再变化，实行early stopping
            if step % max(self.steps // 10, 1) == 0:
                if new_loss.item() > old_loss:
                    return best_attack
                old_loss = new_loss.item()

        return best_attack

    def tanh_space(self, x):  # 对图像进行处理，x=(tanh(w)+1)/2
        return 1 / 2 * (torch.tanh(x) + 1)

    def inverse_tanh_space(self, x):  # 计算w，w=arctan(2*image-1)
        return self.arctanh(torch.clamp(2 * x - 1, min=-1, max=1))

    def arctanh(self, x):  # 求解arctanh
        return 0.5 * torch.log((1 + x) / (1 - x))

    def f(self, output, label):
        tmp = torch.eye(output.shape[1])  # 生成一个10*10、主对角线全为1的矩阵
        tmp = tmp.to(device)
        #  print(tmp)
        one_hot_labels = tmp[label]
        #  print(one_hot_labels)

        #  print(output)
        other = torch.max((1 - one_hot_labels) * output, dim=1)[0]  # 找出除真实类别外概率最大的类别
        real = torch.max(one_hot_labels * output, dim=1)[0]  # 找出真实类别

        return torch.clamp((real - other), min=-self.confidence)  # 无目标攻击，real应该尽可能小，other应该尽可能大