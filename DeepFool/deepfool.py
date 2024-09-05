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
