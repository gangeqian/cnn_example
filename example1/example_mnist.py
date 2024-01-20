import time
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import ipdb



class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # 第一卷积层：输入1通道（灰度图像），输出6通道，卷积核大小为5x5
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        # 第一池化层：最大池化，池化窗口大小为2x2，步幅为2
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 第二卷积层：输入6通道，输出16通道，卷积核大小为5x5
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        # 第二池化层：最大池化，池化窗口大小为2x2，步幅为2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 第一个全连接层：输入维度是16*4*4，输出维度是120
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        # 第二个全连接层：输入维度是120，输出维度是84
        self.fc2 = nn.Linear(120, 84)
        # 第三个全连接层：输入维度是84，输出维度是10，对应10个类别
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 前向传播函数定义网络的数据流向
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x



# 检查GPU是否可用
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 定义数据预处理和加载器
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
test_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)

# 准备模型
input_size = 28*28
output_size = 10
# model = CNN(3,1,1)
model = LeNet5()

model=model.to(device = device,dtype = torch.float32)
# model = torch.nn.Linear(input_size, output_size).to(device)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True).to(device)
# model = torch.compile(model, backend="inductor")

# 定义损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.005)

# 创建CUDA事件
# start_event = torch.cuda.Event(enable_timing=True)
# end_event = torch.cuda.Event(enable_timing=True)

# 记录处理的图片数量和总时间
total_images = 0
epoch_time =  0.0
total_time = 0.0

# 迭代训练
num_epochs = 10

for epoch in range(num_epochs):
    start = time.time()
    for images, labels in tqdm(train_loader):
        # 将数据传输到GPU
        images = images.to(device)
        labels = labels.to(device)

        # 记录开始时间
        # start_event.record()

        # 前向传播
        # images = images.view(-1, input_size)
        output = model(images)
        loss = criterion(output, labels)

        # 反向传播和参数更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 记录结束时间并计算处理时间
        # end_event.record()
        # torch.cuda.synchronize()  # 等待GPU计算完成
        # elapsed_time = start_event.elapsed_time(end_event) / 1000.0  # ms转换为秒

        # 更新统计数据
        total_images += len(images)
        # total_time += elapsed_time

    end = time.time()
    epoch_time = end - start
    total_time += epoch_time
    print(f"epoch {epoch} time:{epoch_time:2f} seconds")
    print('loss:',loss)
# 计算每张图片的平均处理时间
# avg_time_per_image = total_time / total_images

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in tqdm(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))


# 输出结果
print(f"Total images processed: {total_images}")
print(f"Total time taken: {total_time:.2f} seconds")
# print(f"Average time per image: {avg_time_per_image:.5f} seconds")