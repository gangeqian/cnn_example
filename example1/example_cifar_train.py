import torch
import torchvision
import torch.nn as nn
from model import LeNet
import torch.optim as optim
import torchvision.transforms as transforms
import time
import numpy as np
import matplotlib.pyplot as plt

# ref: https://blog.csdn.net/baidu_27066207/article/details/115328376

def main():
    # 对输入的图像数据做预处理
    # 即由shape (H x W x C) in the range [0, 255] → shape (C x H x W) in the range [0.0, 1.0]
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
 
    # 50000张训练图片
    # 第一次使用时要将download设置为True才会自动去下载数据集
    train_set = torchvision.datasets.CIFAR10(root='./data', # 数据集存放目录
                                             train=True,    # 表示是数据集中的训练集
                                             download=True, # 第一次运行时为True，去自动下载数据集，下载完成后改为False
                                             transform=transform) # 预处理过程
    # 加载训练集，实际过程需要分批次（batch）训练
    train_loader = torch.utils.data.DataLoader(train_set,       # 导入的训练集
                                               batch_size=36,   # 每批训练的样本数
                                               shuffle=True,    # 是否随机打乱数据集
                                               num_workers=0)   # 使用线程数，在windows下设置为0
 
    # 10000张验证图片
    # 第一次使用时要将download设置为True才会自动去下载数据集
    val_set = torchvision.datasets.CIFAR10(root='./data', train=False,  # 表示是数据集中的验证集
                                           download=False, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=5000,  # 每批用于验证的样本数量
                                             shuffle=False, num_workers=0)
    # 获取测试集中的图像和标签，用于accuracy计算
    val_data_iter = iter(val_loader)
    val_image, val_label = val_data_iter.next()
 
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
 
    # def imshow(img):
    #     img = img/2+0.5    # unnormalize，反标准化还原回去
    #     npimg = img.numpy()
    #     plt.imshow(np.transpose(npimg, (1,2,0))) # C,H,W ——>H,W,C
    #     plt.show()
 
    net = LeNet()                                       # 定义训练所用的网络模型
    loss_function = nn.CrossEntropyLoss()               # 定义损失函数（这里为交叉熵损失函数）
    optimizer = optim.Adam(net.parameters(), lr=0.001)  # 定义优化器（训练参数，学习率）
 
    for epoch in range(5):  # loop over the dataset multiple times
        # 一个epoch即对整个训练集进行一次训练
        running_loss = 0.0
        time_start = time.perf_counter()
 
        # 遍历训练集，step从0开始计算
        for step, data in enumerate(train_loader, start=0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data   # 获取训练集的图像和标签
            # zero the parameter gradients
            optimizer.zero_grad()   # 历史梯度清零
            # forward + backward + optimize
            outputs = net(inputs)                   # 正向传播
            loss = loss_function(outputs, labels)   # 计算损失
            loss.backward()                         # 反向传播
            optimizer.step()                        # 迭代更新参数
 
            # print statistics（打印耗时、损失、准确率等数据信息）
            running_loss += loss.item()
            if step % 500 == 499:    # print every 500 mini-batches（每500步打印一次）
                with torch.no_grad(): # with是一个上下文管理器
                    # 在这个函数的计算内，都不会改变梯度，即不用计算每个节点的误差损失梯度，防止占用内存及运算资源
                    outputs = net(val_image)  # [batch, 10]测试集传入网络
                    predict_y = torch.max(outputs, dim=1)[1] # 以output中值最大位置（在第一维度）对应的索引（标签）作为预测输出
                    accuracy = torch.eq(predict_y, val_label).sum().item() / val_label.size(0)
 
                    # 打印epoch，step，loss，accuracy
                    print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                          (epoch + 1, step + 1, running_loss / 500, accuracy))
                    # 打印耗时
                    print('%f s' %(time.perf_counter()-time_start))
                    running_loss = 0.0
 
    print('Finished Training')
 
    # 保存训练得到的参数模型
    save_path = './Lenet.pth'
    torch.save(net.state_dict(), save_path)
 
if __name__ == '__main__':
    main()
