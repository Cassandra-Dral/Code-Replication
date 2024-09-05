import numpy as np
import os
import torch
import torchvision
import copy
import torch.autograd
from torch.autograd import Variable
import torch.nn as nn
from matplotlib import pyplot as plt
from PIL import Image
from torchvision.transforms import ToPILImage


# -----------------------定义模型--------------------------
#  定义 3*3 cnn
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


# -----------------------复现DeepFool--------------------------
def deepfool(input, class_num, model, iter_num, overshoot):
    model = model.to(device)
    input = input.to(device)

    #  获取input的真实标签
    output = model.forward(Variable(input[None, :, :, :], requires_grad=True))  # 图像经过网络模型
    output = output.data.cpu().numpy().flatten()  # 将其展平为一维numpy()向量,该向量就是每个类别对应的概率向量
    #  print(np.argmax(output),type(np.argmax(output)))
    prob = (np.array(output)).flatten()  # 将output再展平为一维向量
    prob = prob.argsort()[::-1]  # 按从大到小的顺序排序并建立索引，此时prob[0]就是概率最大的类别标签
    origin_label = prob[0]

    #  进行相关初始化
    input_shape = input.cpu().numpy().shape  # 输入图像的大小
    perturbed_image = copy.deepcopy(input)  # 初始化扰动图像为输入图像
    x = Variable(perturbed_image[None, :], requires_grad=True)
    output_x = model.forward(x)  # x(输入图像)的概率向量

    w = np.zeros(input_shape)
    r_sum = np.zeros(input_shape)

    k_j = origin_label  # 第0次迭代，标签为输入图像的真实标签
    j = 0  # 迭代次数

    while j < iter_num and k_j == origin_label:
        l_0 = np.inf
        output_x[0, prob[0]].backward(retain_graph=True)
        origin_grad = x.grad.data.cpu().numpy().copy()  # 正确分类的梯度,即▽f_k_x0(xj)

        for k in range(1, class_num):  # 这里排除了0，即分类正确的情况
            x.grad.zero_()  # 梯度清零
            output_x[0, prob[k]].backward(retain_graph=True)
            curr_grad = x.grad.data.cpu().numpy().copy()  # 非正确分类的梯度,即▽f_k(xj)

            w_k = curr_grad - origin_grad  # Wk'
            prob_k = output_x[0, prob[k]] - output_x[0, prob[0]]  # fk'
            prob_k = prob_k.data.cpu().numpy()

            l_k = abs(prob_k) / np.linalg.norm(w_k.flatten())
            if l_k < l_0:  # 找到最小的l，该样本距离这个错误分类的超平面最近
                l_0 = l_k
                w = w_k

        r_j = (l_0 + 1e-4) * w / np.linalg.norm(w)  # 更新每一次迭代的扰动
        r_sum = np.float32(r_sum + r_j)  # 更新扰动总和

        #  更新扰动图像
        if torch.cuda.is_available():
            perturbed_image = input + (1 + overshoot) * torch.from_numpy(r_sum).cuda()
        else:
            perturbed_image = input + (1 + overshoot) * torch.from_numpy(r_sum)

        #  更新标签
        x = Variable(perturbed_image, requires_grad=True)
        output_x = model.forward(x)  # x(输入图像)的概率向量
        k_j = np.argmax(output_x.data.cpu().numpy().flatten())

        j += 1

    r_sum = (1 + overshoot) * r_sum

    return r_sum, perturbed_image, k_j  # 返回扰动总和、扰动图像、错误标签


# -----------------------加载数据集与模型--------------------------
dataset = torchvision.datasets.CIFAR10(root="/kaggle/input/dataset1/dataset1", train=True,
                                       transform=torchvision.transforms.ToTensor())
model = torch.load("/kaggle/input/best-model/best_model.pth")
model = model.to(device)
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# -----------------------测试deepfool--------------------------
test_image = dataset[60][0]
copy_image = test_image
copy_image = ToPILImage()(copy_image).convert('RGB')
copy_image.save(os.path.join("/kaggle/working/original_image01.jpg"))

test_image = test_image.to(device)
test_image = torch.reshape(test_image, (1, 3, 32, 32))  # 对图片进行reshape

#  进入评估模式
model.eval()
label = model(test_image)
original_label = labels[label.argmax(1).data.cpu().numpy()[0]]
plt.subplot(1, 2, 1)
plt.imshow(copy_image)
plt.title("original_image")
plt.xlabel(original_label)

#  进行deepfool攻击
test_image = torch.reshape(test_image, (3, 32, 32))  # 对图片进行reshape
r_sum, perturbed_image, perturbed_label = deepfool(test_image, 10, model, 50, 0.02)

perturbed_image = torch.reshape(perturbed_image, (3, 32, 32))
PIL_perturbed_image = ToPILImage()(perturbed_image).convert('RGB')
PIL_perturbed_image.save(os.path.join("/kaggle/working/perturbed_image01.jpg"))
plt.subplot(1, 2, 2)
plt.imshow(PIL_perturbed_image)
plt.title("perturbed_image")
plt.xlabel(labels[perturbed_label])
plt.show()