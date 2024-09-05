import numpy as np
import os
from __future__ import absolute_import, division, print_function
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

model = torch.load("/kaggle/input/best-model/best_model.pth")
model = model.to(device)

# -------------------HopSkipJumpAttack，默认采用l2距离----------------------
c_min = 0.0  # 图像下界
c_max = 1.0  # 图像上界
constraint = 'l2'  # 使用l2范数
T = 40  # 迭代次数
gamma = 1.0  # 用于设置theta, constraint=l2, theta=gamma*d^(-3/2), 否则theta=gamma*d^(-2)
target_image = None  # 无目标攻击
target_label = None
shape = None
max_num_evals = 10000
init_num_evals = 100
origin_label = None


#  剪裁图像，限制图像在[c_min,c_max]的范围内
def clip_image(input):
    output = np.maximum(input, c_min)
    output = np.minimum(output, c_max)
    return output


#  输出分类结果，1表示攻击成功，0表示攻击失败
def classifier(model, input):
    input = clip_image(input)  # 剪裁图像

    copy_input = copy.deepcopy(input)  # 复制输入图像用于预测其分类
    copy_input = torch.tensor(copy_input)
    copy_input = copy_input.type(torch.FloatTensor)
    copy_input = torch.reshape(copy_input, (1, 3, 32, 32))
    copy_input = copy_input.to(device)

    prob = model(copy_input)
    prob = prob.data.cpu().numpy().flatten()
    label = np.argmax(prob)  # 获取经过网络模型的分类，label是numpy.int64

    if target_image is None:  # 无目标攻击
        return label != origin_label
    else:  # 有目标攻击
        return label == target_label


#  初始化图像
def init(model, input):
    i = 0
    suc = 0
    low = 0.0
    high = 1.0
    noise = np.zeros(shape)

    if target_image is None:  # 无目标攻击
        #  随机生成噪声作为分类错误的样本
        while True:
            rand_noise = np.random.uniform(c_min, c_max, size=shape)
            suc = classifier(model, rand_noise)  # 判断该噪声能否攻击成功
            i += 1;

            if (suc):  # 攻击成功
                noise = rand_noise
                break
            if i > 1e4:  # 攻击失败
                print("init failed!\n")

        #  二分查找到距离原样本l2距离最近的样本
        while high - low > 0.001:
            mid = (high + low) / 2.0
            blended = mid * noise + (1 - mid) * input  # 混合了噪声和原样本的样本
            suc = classifier(model, blended[None])

            if suc:  # 距离噪声过近，mid应该减小
                high = mid
            else:  # 距离原样本过近，mid应该增大
                low = mid

        initial = high * noise + (1 - high) * input  # 返回混合样本

    else:  # 有目标攻击
        initial = target_image  # 返回目标图像

    return initial


#  计算两个样本之间的l2或l∞距离
def distance(origin_image, pert_image):
    if constraint == 'l2':
        return np.linalg.norm(origin_image - pert_image)
    else:
        return np.max(abs(origin_image - pert_image))


#  判断在给定的步长下攻击能否成功
def succeed(stepsize, x_t, v_t):
    new_x_t = x_t + stepsize * v_t
    suc = classifier(model, new_x_t[None])
    return suc


#  确定步长
def cal_stepsize(model, dis, x_t, v_t, cur_iter):
    stepsize = dis / np.sqrt(cur_iter)  # 计算步长

    while not succeed(stepsize, x_t, v_t):  # 攻击失败则步长减半
        stepsize /= 2.0

    return stepsize


#  确定扰动参数
def cal_perturbation_para(dis_list, cur_iter, theta, d):
    if cur_iter == 1:
        perturbation_para = 0.1 * (c_max - c_min)
    else:
        if constraint == 'l2':
            perturbation_para = np.sqrt(d) * theta * dis_list
        else:
            perturbation_para = d * theta * dis_list

    return perturbation_para


#  lp投影，返回基于原始图像和扰动图像的混合图像
def lp_projection(origin_image, pert_images, alphas):  # 扰动图像是一个集合
    #  确定α的大小
    alphas_shape = [len(alphas)] + [1] * len(shape)
    alphas = alphas.reshape(alphas_shape)

    if constraint == 'l2':
        return alphas * pert_images + (1 - alphas) * origin_image
    else:
        #     outputs = np.maximum(input, c_min)
        #     outputs = np.minimum(output, c_max)
        #     outputs = minimum(pert_images, origin_image + alphas)
        #     outputs = maximum(output, origin_image - alphas)
        #     此处存疑，如果按照代码来就和论文不一样
        outputs = np.maximum(pert_images, origin_image - alphas)
        outputs = np.minimum(outputs, origin_image + alphas)
        return outputs


#  二分搜索逼近边界
def bin_search(model, origin_image, pert_images, t):  # 扰动图像是一个集合
    #  计算每个扰动图像与原始图像的距离
    dis_list = np.array([distance(origin_image, pert_image) for pert_image in pert_images])

    if constraint == 'l2':
        highs = np.ones(len(pert_images))  # 置上界为1
        threshold = t
    else:
        highs = dis_list  # 置上界为dis_list
        threshold = np.minimum(dis_list * theta, theta)

    lows = np.zeros(len(pert_images))  # 置下界为0

    #  二分搜索，确定上界α_u
    while np.max((highs - lows) / threshold) > 1:  # |highs- lows| > threshold

        mids = (highs + lows) / 2.0
        mids_images = lp_projection(origin_image, pert_images, mids)
        suc = classifier(model, mids_images)

        if suc:
            highs = mids
        else:
            lows = mids

    output_images = lp_projection(origin_image, pert_images, highs)

    distances = np.array([distance(origin_image, output_image) for output_image in output_images])  # 计算每个扰动图像与原始图像的距离

    idx = np.argmin(distances)  # 找到与原样本距离最短的新样本的下标
    return output_images[idx], dis_list[idx]  # 返回新样本与距离，这里存疑


#  梯度估计
def approximate_grad(model, input, num_evals, perturbation_para):
    #  随机生成向量
    noise_shape = [num_evals] + list(shape)
    if constraint == 'l2':
        random_v = np.random.randn(shape[0], 1)
    else:
        random_v = np.random.uniform(low=-1, high=1, size=noise_shape)

    random_v = random_v / np.sqrt(np.sum(random_v ** 2, keepdims=True))  # u_b, 分子是向量，分母是该向量的l2范数

    pert = input + perturbation_para * random_v
    pert = clip_image(pert)

    random_v = (pert - input) / perturbation_para

    #  查询模型
    decisions = classifier(model, pert)
    fv = 2 * decisions - 1.0  # fai_x*(x_t+δ*u_b)

    if np.mean(fv) == 1.0:  # 攻击都成功
        grad_f = np.mean(random_v, axis=0)
    elif np.mean(fv) == -1.0:  # 攻击都失败
        grad_f = - np.mean(random_v, axis=0)
    else:  # 其他情况
        fv -= np.mean(fv)  # 减去均值
        grad_f = np.mean(fv * random_v, axis=0)

    v = grad_f / np.linalg.norm(grad_f)

    return v


#  HopSkipJumpAttack攻击算法
def hopskipjumpattack(model, input):
    #  输入图像shape
    shape = input.shape

    #  输入图像尺寸的乘积
    d = int(np.prod(input.shape))

    #  设置theta
    if constraint == 'l2':
        theta = gamma / (np.sqrt(d) * d)
    else:
        theta = gamma / (d * d)

    #  初始化图像
    pert_image = init(model, input)

    #  计算距离
    pert_image, dis_list = bin_search(model, input, np.expand_dims(pert_image, 0), theta)
    d0 = distance(input, pert_image)  # x0与x*

    #  开始迭代
    for t in np.arange(T):

        #  选择扰动参数
        perturbation_para = cal_perturbation_para(dis_list, t + 1, theta, d)

        #  选择评估轮数
        num_evals = int(init_num_evals * np.sqrt(t + 1))
        if num_evals > max_num_evals:
            num_evals = max_num_evals

        #  梯度估计
        v = approximate_grad(model, pert_image, num_evals, perturbation_para)
        if constraint == 'l2':
            v_t = v
        else:
            v_t = np.sign(v)

        #  确定步长
        stepsize = cal_stepsize(model, d0, pert_image, v_t, t + 1)

        #  x_t + 步长 * v_t
        pert_image = clip_image(pert_image + stepsize * v_t)

        #  二分查找
        pert_image, dislist = bin_search(model, input, pert_image[None], theta)

        #  更新距离
        d0 = distance(input, pert_image)

    return pert_image


# -------------------测试----------------------
#  准备数据
dataset = torchvision.datasets.CIFAR10(root="/kaggle/input/dataset1/dataset1", train=True,
                                       transform=torchvision.transforms.ToTensor())
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
test_image = dataset[300][0]
copy_image = test_image
copy_image = ToPILImage()(copy_image).convert('RGB')
copy_image.save(os.path.join("/kaggle/working/original_image02.jpg"))
test_image = test_image.to(device)

#  进入评估模式
model.eval()

output = model.forward(Variable(test_image[None, :, :, :]))  # 图像经过网络模型
output = output.data.cpu().numpy().flatten()  # 将其展平为一维numpy()向量,该向量就是每个类别对应的概率向量

prob = (np.array(output)).flatten()  # 将output再展平为一维向量
prob = prob.argsort()[::-1]  # 按从大到小的顺序排序并建立索引，此时prob[0]就是概率最大的类别标签
origin_label = prob[0]

plt.subplot(1, 2, 1)
plt.imshow(copy_image)
plt.title("original_image")
plt.xlabel(labels[origin_label])

test_image = torch.reshape(test_image, (1, 3, 32, 32))
test_image = test_image.data.cpu().numpy()  # 输入图像是ndarray类型
shape = test_image.shape

pert_image = hopskipjumpattack(model, test_image)  # 进行攻击
pert_image = torch.tensor(pert_image)
pert_image = pert_image.type(torch.FloatTensor)
pert_image = torch.reshape(pert_image, (3, 32, 32))
PIL_pert_image = ToPILImage()(pert_image).convert('RGB')
PIL_pert_image.save(os.path.join("/kaggle/working/perturbed_image02.jpg"))

pert_image = pert_image.to(device)
pert_output = model.forward(Variable(pert_image[None, :, :, :]))  # 图像经过网络模型
pert_output = pert_output.data.cpu().numpy().flatten()  # 将其展平为一维numpy()向量,该向量就是每个类别对应的概率向量

pert_prob = (np.array(pert_output)).flatten()  # 将output再展平为一维向量
pert_prob = pert_prob.argsort()[::-1]  # 按从大到小的顺序排序并建立索引，此时prob[0]就是概率最大的类别标签
pert_label = pert_prob[0]

plt.subplot(1, 2, 2)
plt.imshow(PIL_pert_image)
plt.title("perturbed_image")
plt.xlabel(labels[pert_label])
plt.show()