'''
项目适配
'''

import json
import re
from xml.dom import minidom
import os
from utils.datasets import *

project_dir = os.path.dirname(__file__)

config_dir = os.path.join(project_dir, "config")

xml_dir = os.path.join(project_dir, "train_labels")

def generator_cfg(classes):
    filter_nums = 3 * (classes + 5)
    writer = open(os.path.join(config_dir, 'yolov3-cola.cfg'), 'w')
    with open(os.path.join(config_dir, "model.cfg"), 'r') as f:
        lines = f.readlines()
        for line in lines:
            # print(line)
            if re.search('\\$NUM_CLASSES', line):
                print("find ==>> num classes")
                line = line.replace('$NUM_CLASSES', str(classes))
            elif re.search('\\$NUM_FILTERS', line):
                print("find ==>> num filters")
                line = line.replace('$NUM_FILTERS', str(filter_nums))
            writer.write(line)
    writer.close()

def showdata(loader):
    dataiter = iter(loader)
    _, images, labels = dataiter.next()
    print(len(images))
    print(labels)

def coco_dataset_read():
    dataset = CocoDetection("F:\data\coco2014\\train2014", "F:\data\coco2014\\annotations\instances_train2014.json")
    trainloader = torch.utils.data.DataLoader(dataset,  batch_size=1, shuffle=False, num_workers=2)
    showdata(trainloader)

START_BOUNDING_BOX_ID = 1
PRE_DEFINE_CATEGORIES = {'coca':1, 'pepsi':2}

# If necessary, pre-define category and its id
#  PRE_DEFINE_CATEGORIES = {"aeroplane": 1, "bicycle": 2, "bird": 3, "boat": 4,
                         #  "bottle":5, "bus": 6, "car": 7, "cat": 8, "chair": 9,
                         #  "cow": 10, "diningtable": 11, "dog": 12, "horse": 13,
                         #  "motorbike": 14, "person": 15, "pottedplant": 16,
                         #  "sheep": 17, "sofa": 18, "train": 19, "tvmonitor": 20}


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

def get(root, name):
    vars = root.findall(name)
    return vars


def get_and_check(root, name, data=False):
    node = root.getElementsByTagName(name)
    if data:
        return node[0].firstChild.data
    else:
        return node

# def get_and_check(root, name, length):
#     vars = root.findall(name)
#     if len(vars) == 0:
#         raise NotImplementedError('Can not find %s in %s.'%(name, root.tag))
#     if length > 0 and len(vars) != length:
#         raise NotImplementedError('The size of %s is supposed to be %d, but is %d.'%(name, length, len(vars)))
#     if length == 1:
#         vars = vars[0]
#     return vars


def get_filename_as_int(filename):
    try:
        filename = os.path.splitext(filename)[0]
        return int(filename)
    except:
        raise NotImplementedError('Filename %s is supposed to be an integer.'%(filename))

'''
    :param xml_list: 需要转换的XML文件列表
    :param xml_dir: XML的存储文件夹
    :param json_file: 导出json文件的路径
    :return: None
'''
def convert(xml_list, xml_dir, json_file):
    list_fp = xml_list
    json_dict = {"images":[], "type": "instances", "annotations": [],
                 "categories": []}
    categories = PRE_DEFINE_CATEGORIES
    bnd_id = START_BOUNDING_BOX_ID
    for id, line in enumerate(list_fp):
        line = line.strip()
        print("Processing %s"%(line))
        xml_f = os.path.join(xml_dir, line)
        print(xml_f)
        labelXML = minidom.parse(xml_f)
        filename = get_and_check(labelXML, 'filename', True)
        print(filename)

        ## The filename must be a number
        size = get_and_check(labelXML, 'size')[0]
        width = round(float(get_and_check(size, 'width', True)))
        height = round(float(get_and_check(size, 'height', True)))
        image = {'file_name': filename, 'height': height, 'width': width,
                 'id': id}
        json_dict['images'].append(image)
        ## Cruuently we do not support segmentation
        #  segmented = get_and_check(root, 'segmented', 1).text
        #  assert segmented == '0'
        for obj in get_and_check(labelXML, 'object'):
            category = get_and_check(obj, 'name', True)
            if category not in categories:
                new_id = len(categories)
                categories[category] = new_id

            category_id = categories[category]
            bndbox = get_and_check(obj, 'bndbox')[0]
            xmin = round(float(get_and_check(bndbox, 'xmin', True))) - 1
            ymin = round(float(get_and_check(bndbox, 'ymin', True))) - 1
            xmax = round(float(get_and_check(bndbox, 'xmax', True)))
            ymax = round(float(get_and_check(bndbox, 'ymax', True)))

            o_width = abs(xmax - xmin)
            o_height = abs(ymax - ymin)
            ann = {'area': o_width * o_height, 'iscrowd': 0, 'image_id': id,
                   'bbox': [xmin, ymin, o_width, o_height],
                   'category_id': category_id, 'id': bnd_id, 'ignore': 0,
                   'segmentation': []}
            json_dict['annotations'].append(ann)
            bnd_id = bnd_id + 1

    for cate, cid in categories.items():
        cat = {'supercategory': 'none', 'id': cid, 'name': cate}
        json_dict['categories'].append(cat)
    json_fp = open(json_file, 'w')
    json_str = json.dumps(json_dict)
    json_fp.write(json_str)
    json_fp.close()

def transform():
    xml_list = []
    with open(os.path.join(project_dir, 'list.labels_txt'), 'r') as f:
        lines = f.readlines()
        for id, line in enumerate(lines):
            if id >= 87330:
                break
            file_name = line.split("/")[-1]
            xml_list.append(file_name)
    print("标签文件个数: {}".format(len(xml_list)))
    convert(xml_list, xml_dir, os.path.join(project_dir, 'train_labels.json'))

if __name__ == '__main__':
    coco_dataset_read()