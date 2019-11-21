from __future__ import print_function
import torch
from pycocotools.coco import COCO
from skimage import io
def gpu_test():
    if torch.cuda.is_available():
        cuda = torch.device('cuda')
        y = torch.ones(5, 4, device=cuda)
        print(y)
        print(y.to('cpu', torch.double))

def gradients():
    x = torch.ones(2, 2, requires_grad=True)
    y = x + 2
    z = y * y * 3
    out = z.mean()
    # print(z, out)  # 会有一个 grad_fn
    # out.backward() # out 对x求梯度，因为out变量是个标量，所以不需要指定 gradients 参数
    # print(x.grad)
    z.backward(torch.Tensor([[1., 1.], [1., 1.]])) # z 是个向量，需要指定参数
    print(x.grad)

def showImg():
    coco = COCO('F:\data\coco2014\\annotations\instances_train2014.json')
    image_url = coco.loadImgs(9)[0]['coco_url']
    image = io.imread(image_url)
    io.imshow(image)
    io.show()

if __name__ == '__main__':
    showImg()

