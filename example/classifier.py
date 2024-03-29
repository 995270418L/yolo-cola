import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim  # loss function package

import matplotlib.pyplot as plt
import numpy as np

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=False, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=False, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=True, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

PATH = './model.pth'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Classfier(nn.Module):

    def __init__(self):
        super(Classfier, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def showdata(loader):
    dataiter = iter(loader)
    images, labels = dataiter.next()
    print(len(images))
    imshow(torchvision.utils.make_grid(images))
    print(''.join("%s\t" % classes[labels[j]] for j in range(4)))
    return images

def train_model_then_save():
    net = Classfier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    print("start training ...")
    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(trainloader):  # 每次提取4张图片训练, 一共有 50000 张照片在训练集
            inputs, labels = data
            optimizer.zero_grad()  # 每次重置 gradients

            outputs = net(inputs)  # 4 * 3 * 32 * 32
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()  # update weight

            running_loss += loss.item()
            if i % 2000 == 1999:
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    print("Finished Training")
    print("save model")

    torch.save(net.state_dict(), PATH)

def test_model():
    net = Classfier()
    net.load_state_dict(torch.load(PATH))
    images = showdata(testloader)
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)  # 返回每一行中的最大值, 第一个是值，第二个是索引
    print('Preticted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

def test_accuracy():
    net = Classfier()
    net.load_state_dict(torch.load(PATH))
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))

if __name__ == '__main__':
    test_accuracy()