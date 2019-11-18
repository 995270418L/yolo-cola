import os
import random
import xml.etree.ElementTree as ET

trainval_percent = 0.1
train_percent = 0.9
sets = ['train_label', 'valid_label']
classes = ["coca", 'pepsi']

xmlfilepath = 'data/cola/labels'

# 根据标签 生成txt描述文件
def transform_txt():
    total_xml = os.listdir(xmlfilepath)
    num = len(total_xml)
    list = range(num)
    tv = int(num * trainval_percent)
    tr = int(tv * train_percent)
    trainval = random.sample(list, tv)  # 少数数据
    train = random.sample(trainval, tr) # 多数数据

    train_txt = open('data/cola/train_label.txt', 'w')
    valid_txt = open('data/cola/valid_label.txt', 'w')

    for i in list:
        name = total_xml[i].split('.')[0] + ".jpg"
        if i in trainval:
            valid_txt.write(name + os.linesep)
            if i in train:
                train_txt.write(name + os.linesep)
        else:
            train_txt.write(name + os.linesep)
    train_txt.close()
    valid_txt.close()

def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def convert_annotation(image_id):
    print("Process id :{}".format(image_id))
    in_file = open('data/cola/labels/%s.xml' % (image_id), encoding='utf-8')
    out_file = open('data/cola/labels_txt/%s.txt' % (image_id), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls not in classes:
            print('类别错误')
            os._exit(3)
            return
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (round(float(xmlbox.find('xmin').text), 3), round(float(xmlbox.find('xmax').text), 3), round(float(xmlbox.find('ymin').text), 3),
             round(float(xmlbox.find('ymax').text), 3))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + os.linesep)
    out_file.close()

# 每个 jpg 生成对应的 txt 文件
def txt_values():
    for image_set in sets:
        image_ids = open('data/cola/%s.txt' % (image_set)).read().strip().split()
        for image_id in image_ids:
            convert_annotation(image_id.split('.')[0])

def process():
    transform_txt()
    txt_values()

if __name__ == '__main__':
    process()