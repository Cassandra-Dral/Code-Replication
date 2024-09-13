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

# -------------------定义模型----------------------
# 定义 3*3 cnn
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if (self.downsample):
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


#  定义 ResNet-18
class ResNet(torch.nn.Module):
    def __init__(self, block, layers, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = torch.nn.BatchNorm2d(16)
        self.relu = torch.nn.ReLU(inplace=True)
        self.layer1 = self._make_layers(block, 16, layers[0])
        self.layer2 = self._make_layers(block, 32, layers[1], 2)
        self.layer3 = self._make_layers(block, 64, layers[2], 2)
        self.layer4 = self._make_layers(block, 128, layers[3], 2)
        self.avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(128, num_classes)

    def _make_layers(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = torch.nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                torch.nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


#  指定设备类型
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


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


# -----------------------测试L2攻击--------------------------
dataset = torchvision.datasets.CIFAR10(root="/kaggle/input/dataset1/dataset1", train=True,
                                       transform=torchvision.transforms.ToTensor())
model = torch.load("/kaggle/input/best-model/best_model.pth")
model = model.to(device)
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

test_image = dataset[460][0]
copy_image = test_image
copy_image = ToPILImage()(copy_image).convert('RGB')
copy_image.save(os.path.join("/kaggle/working/original_image02.jpg"))

test_image = test_image.to(device)
test_image = torch.reshape(test_image, (1, 3, 32, 32))  # 对图片进行reshape
copy_image1 = test_image
copy_image1 = copy_image1.to(device)

#  进入评估模式
model.eval()
label = model(test_image)
copy_label = torch.tensor([label.argmax(1).data.cpu().numpy()[0]]).to(device)
original_label = labels[label.argmax(1).data.cpu().numpy()[0]]
plt.subplot(1, 2, 1)
plt.imshow(copy_image)
plt.title("original_image")
plt.xlabel(original_label)

test_image = torch.reshape(test_image, (1, 3, 32, 32))

attack1 = L2_attack(model)  # 进行攻击
adv_image = attack1.forward(copy_image1, copy_label)
adv_image = torch.tensor(adv_image)
adv_image = adv_image.type(torch.FloatTensor)
adv_image = torch.reshape(adv_image, (3, 32, 32))
PIL_adv_image = ToPILImage()(adv_image).convert('RGB')
PIL_adv_image.save(os.path.join("/kaggle/working/adv_image02.jpg"))

adv_image = adv_image.to(device)
adv_output = model.forward(Variable(adv_image[None, :, :, :]))  # 图像经过网络模型
adv_output = adv_output.data.cpu().numpy().flatten()  # 将其展平为一维numpy()向量,该向量就是每个类别对应的概率向量

adv_prob = (np.array(adv_output)).flatten()  # 将output再展平为一维向量
adv_prob = adv_prob.argsort()[::-1]  # 按从大到小的顺序排序并建立索引，此时prob[0]就是概率最大的类别标签
adv_label = adv_prob[0]

plt.subplot(1, 2, 2)
plt.imshow(PIL_adv_image)
plt.title("adversarial_image")
plt.xlabel(labels[adv_label])
plt.show()