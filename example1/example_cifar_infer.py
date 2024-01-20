
import torch
import torchvision.transforms as transforms
from PIL import Image
from model import LeNet
 
def main():
    transform = transforms.Compose(
        [transforms.Resize((32, 32)),  # 首先需将数据集resize成与训练集图像一样的大小
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
 
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
 
    # 实例化网络
    net = LeNet()
    # 用下述函数载入刚刚训练好的网络模型
    net.load_state_dict(torch.load('Lenet.pth'))
 
    # 导入要测试的图像，用PIL载入
    #im = Image.open('img/dog1.jpeg')
    #im = Image.open('img/cat1.jpeg')
    #im = Image.open('img/plane2.jpeg')
    im = Image.open('img/ship1.jpg')
    im = transform(im)  # [C, H, W]
    # 对数据增加一个新维度，因为tensor的参数是[batch, channel, height, width]
    im = torch.unsqueeze(im, dim=0)  # [N, C, H, W]
 
    with torch.no_grad():
        outputs = net(im)
        predict = torch.max(outputs, dim=1)[1].data.numpy()
        # 预测结果也可用softmax，输出十个概率，输出结果中最大概率值对应的索引即为预测标签的索引
        # predict = torch.softmax(outputs, dim=1)
    print("class:",classes[int(predict)])
 
 
if __name__ == '__main__':
    main()