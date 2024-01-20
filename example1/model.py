import torch.nn as nn
import torch.nn.functional as F
# torch.nn包用来构建神经网络
 
class LeNet(nn.Module):                     #定义一个模型，继承于nn.Module父类
    def __init__(self):
        super(LeNet, self).__init__()       # 多继承，调用父类函数
        self.conv1 = nn.Conv2d(3, 16, 5)    # 输入深度，卷积核个数，卷积核尺寸
        self.pool1 = nn.MaxPool2d(2, 2)     # 池化核尺寸为2*2
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
 
    def forward(self, x):       # 输入数据，Tensor
        x = F.relu(self.conv1(x))    # input(3, 32, 32) output(16, 28, 28)
        x = self.pool1(x)            # output(16, 14, 14)
        x = F.relu(self.conv2(x))    # output(32, 10, 10)
        x = self.pool2(x)            # output(32, 5, 5)
        x = x.view(-1, 32*5*5)       # output(32*5*5)   展平操作，-1表示第一维度推理
        x = F.relu(self.fc1(x))      # output(120)
        x = F.relu(self.fc2(x))      # output(84)
        x = self.fc3(x)              # output(10) 
        # 最后一层为什么没有用softmax函数，内部已经有实现
        return x