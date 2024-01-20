import time
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
import os
from torch import _dynamo 

# torch._dynamo.config.verbose = True
# torch._dynamo.config.suppress_errors = True

#卷积模块，由卷积核和激活函数组成
class conv_block(nn.Module):
    def __init__(self,ks,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=ks,stride=1,padding=1,bias=True),  #二维卷积核，用于提取局部的图像信息
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True), #这里用ReLU作为激活函数
            nn.Conv2d(ch_out, ch_out, kernel_size=ks,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )
    def forward(self,x):
        return self.conv(x)

class CNN(nn.Module):
    def __init__(self,kernel_size,in_ch,out_ch):
        super(CNN, self).__init__()
        feature_list = [16,32,64,128,256]   #代表每一层网络的特征数，扩大特征空间有助于挖掘更多的局部信息
        self.conv1 = conv_block(kernel_size,in_ch,feature_list[0])
        self.conv2 = conv_block(kernel_size,feature_list[0],feature_list[1])
        self.conv3 = conv_block(kernel_size,feature_list[1],feature_list[2])
        self.conv4 = conv_block(kernel_size,feature_list[2],feature_list[3])
        self.conv5 = conv_block(kernel_size,feature_list[3],feature_list[4])
        self.fc =  nn.Sequential(           #全连接层主要用来进行分类，整合采集的局部信息以及全局信息
            nn.Linear(feature_list[4] * 28 * 28, 1024),  #此处28为MINST一张图片的维度
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
 
    def forward(self,x):
        device = x.device
        x1 = self.conv1(x )
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x5 = x5.view(x5.size()[0], -1)  #全连接层相当于做了矩阵乘法，所以这里需要将维度降维来实现矩阵的运算
        out = self.fc(x5)
        return out



parser = argparse.ArgumentParser()
if (torch.__version__).startswith('2.0'):
    parser.add_argument("--local-rank", type=int, required=False)
else:
    parser.add_argument("--local_rank", type=int, required=False)
parser.add_argument("--gpus", type=str, required=True)
parser.add_argument("--ngpus", default=1, type=int)
args = parser.parse_args()
# local_rank = os.environ['LOCAL_RANK']
local_rank = args.local_rank
if args.gpus == "all":
    gpus = [str(i) for i in range(args.ngpus)]
else:
    gpus = args.gpus.split(",")
# os.environ["CUDA_VISIBLE_DEVIECES"]=str(gpus[local_rank])
torch.cuda.set_device(local_rank)  # 等价？
## DDP 初始化
dist.init_process_group(backend='nccl')


# 检查GPU是否可用
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# 定义数据预处理和加载器
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
ddp_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
train_loader = DataLoader(train_dataset, 
                          batch_size=64, 
                          num_workers=2, 
                          pin_memory=True,
                          drop_last= True,
                        sampler=ddp_sampler)


# 准备模型
input_size = 28*28
output_size = 10
model = CNN(3,1,1).cuda()
# model = torch.nn.Linear(input_size, output_size).to(device)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True).to(device)

ckpt_path = None
if dist.get_rank() == 0 and ckpt_path is not None:
    model.load_state_dict(torch.load(ckpt_path))


process_group = torch.distributed.new_group(range(dist.get_world_size()))


# DDP: 构造DDP model
# model = DDP(model, device_ids=[local_rank], output_device=local_rank) 
# model = DDP(model, find_unused_parameters=True) 
model = DDP(model)

model = nn.SyncBatchNorm.convert_sync_batchnorm(
    model, process_group
)

# pytorch2 compile
# model = torch.compile(model, backend="inductor")


# 定义损失函数和优化器
criterion = torch.nn.CrossEntropyLoss().cuda() #?? 需要GPU??
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)


# 记录处理的图片数量和总时间
total_images = 0
epoch_time =  0.0
total_time = 0.0

# 迭代训练
model.train()
num_epochs = 10
for epoch in tqdm(range(num_epochs)):
    start = time.time()
    #DistributeSample 通过set_epoch来shuffle
    train_loader.sampler.set_epoch(epoch)
    for step,(images, labels) in tqdm(enumerate(train_loader)):
        # 将数据传输到GPU
        images = images.cuda()
        labels = labels.cuda()
        # 前向传播
        output = model(images)
        loss = criterion(output, labels)
        # 反向传播和参数更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 更新统计数据
        total_images += len(images)
    print(f'loss:{loss.cpu():.4f}')
    end = time.time()
    epoch_time = end - start
    total_time += epoch_time
    print(f"epoch {epoch} time:{epoch_time:2f} seconds")
    
    if dist.get_rank() == 0:
        torch.save(model.module.state_dict(),"%d.ckpt"%epoch)

# 输出结果
print(f"Total images processed: {total_images}")
print(f"Total time taken: {total_time:.2f} seconds")
