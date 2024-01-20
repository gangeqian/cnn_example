import time
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# 检查GPU是否可用
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 定义数据预处理和加载器
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)

# 准备模型
input_size = 28*28
output_size = 10
model = torch.nn.Linear(input_size, output_size).to(device)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True).to(device)
# model = torch.compile(model, backend="inductor")

# 定义损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 创建CUDA事件
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

# 记录处理的图片数量和总时间
total_images = 0
total_time = 0.0

# 迭代训练
num_epochs = 20
for epoch in tqdm(range(num_epochs)):
    for images, labels in train_loader:
        # 将数据传输到GPU
        images = images.to(device)
        labels = labels.to(device)

        # 记录开始时间
        start_event.record()

        # 前向传播
        images = images.view(-1, input_size)
        output = model(images)
        loss = criterion(output, labels)

        # 反向传播和参数更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 记录结束时间并计算处理时间
        end_event.record()
        torch.cuda.synchronize()  # 等待GPU计算完成
        elapsed_time = start_event.elapsed_time(end_event) / 1000.0  # ms转换为秒

        # 更新统计数据
        total_images += len(images)
        total_time += elapsed_time

# 计算每张图片的平均处理时间
avg_time_per_image = total_time / total_images

# 输出结果
print(f"Total images processed: {total_images}")
print(f"Total time taken: {total_time:.2f} seconds")
print(f"Average time per image: {avg_time_per_image:.5f} seconds")