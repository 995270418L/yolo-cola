# import matplotlib as plt
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np

unloader = transforms.ToPILImage()
# 读取图片操作
def init(path='data/test.jpg'):
    img_obj = Image.open(path).convert('RGB')
    print(img_obj.size) # 1333 * 750 只会返回双通道
    print(img_obj.mode) # L 为灰度图， RGB 为真彩色 RGBA 为加了透明通道
    # img_obj.show()
    img = transforms.ToTensor()(img_obj)
    return img

'''
图片方形化处理
'''
def square(img):
    c, h, w = img.shape
    print("channel : %d, height: %d, width: %d", c, h, w)
    dim = np.abs(h - w)
    pad1, pad2 = dim // 2, dim - dim // 2
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    img = F.pad(img, pad, 'constant', 10)
    print("square shape: {}".format(img.shape))
    # img_pil = unloader(img)
    # img_pil.show()
    return img, pad

'''
采样
'''
def sampling(img, target_size=416):
    img = F.interpolate(img.unsqueeze(0), size=target_size, mode='nearest').squeeze(0)
    print("sampling size: {}".format(img.shape))
    unloader(img).show()
    return img

def run():
    img = init()
    img, _ = square(img)
    sampling(img)

if __name__ == '__main__':
    run()