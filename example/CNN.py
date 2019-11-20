import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

'''
CNN 设计层数:
    1:  输入层
    2： 卷积层 * ( 3*3, strides=1, 
    3： 激励层 *
    4： 池化层 *
    5： 全连接层 *
    6： 输出层
'''
class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential()
        # S1  该层为卷积层,卷积核大小是3x3,激活函数为RELU.输入一个28x28的矩阵,输出32个28x28 维的feature map
        self.conv1.add_module('1_convolution_layer', nn.Conv2d(1, 32, kernel_size=3))
        self.conv1.add_module('1_activate_layer', nn.ReLU())
        self.conv1.add_module('1_pooled_layer', nn.MaxPool2d(2))

        self.conv2 = nn.Sequential()
        # 该层为卷积层,卷积核大小是3x3,激活函数为RELU.输入32个14x14的feature map,输出64个14x14 的feature map
        self.conv2.add_module('2_convolution_layer', nn.Conv2d(32, 64, kernel_size=3))
        self.conv2.add_module('2_activate_layer', nn.ReLU())
        self.conv2.add_module('2_pooled_layer', nn.MaxPool2d(2))

        self.conv3 = nn.Sequential()
        # 该层为卷积层,卷积核大小是3x3,激活函数为RELU.输入64个7x7的feature map,输出128个7x7 的feature map
        self.conv3.add_module('3__convolution_layer', nn.Conv2d(64, 128, kernel_size=3))
        self.conv3.add_module('3_activate_layer', nn.ReLU())
        self.conv3.add_module('3_pooled_layer', nn.MaxPool2d(2))

        # 128 * 4 * 4
        self.out = nn.Linear(128 * 4 * 4, 10) # 10 个输出

    def forward(self, x):
        # ? * 28 * 28 * 3
        print(x.shape())
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        print(x.shape())
        x = x.view(x.size(0), -1)
        print(x.shape())
        output = self.out(x)
        return output

cnn = CNN()
print(cnn)

optimizer = torch.optim.Adam(cnn.parameters, lr=1e-3)
loss_func = nn.CrossEntropyLoss()

dataset = torchvision.datasets.MNIST("./data", train=True, download=True, transform=torchvision.transforms.ToTensor)
trainloader = torch.utils.data.DataLoader(dataset, )