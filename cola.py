'''
项目适配
'''

import os
import re

project_dir = os.path.dirname(__file__)
config_dir = os.path.join(project_dir, "config")

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

if __name__ == '__main__':
    generator_cfg(2)