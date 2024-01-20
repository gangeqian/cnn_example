import time
import torch
from torchvision import datasets, transforms
from tqdm import tqdm

# 定义数据预处理和加载器
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# 准备模型
input_size = 28*28
output_size = 10
model = torch.nn.Linear(input_size, output_size)

# 定义损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 记录处理的图片数量和总时间
total_images = 0
total_time = 0.0

# 迭代训练
num_epochs = 20
for epoch in tqdm(range(num_epochs)):
    for images, labels in train_loader:
        # 记录开始时间
        start_time = time.time()

        # 前向传播
        images = images.view(-1, input_size)
        output = model(images)
        loss = criterion(output, labels)

        # 反向传播和参数更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 记录结束时间并计算处理的时间
        end_time = time.time()
        processing_time = end_time - start_time

        # 更新统计数据
        total_images += len(images)
        total_time += processing_time

# 计算每张图片的平均处理时间
avg_time_per_image = total_time / total_images

# 输出结果
print(f"Total images processed: {total_images}")
print(f"Total time taken: {total_time:.2f} seconds")
print(f"Average time per image: {avg_time_per_image:.5f} seconds")