from collections import defaultdict
import os
import xml.etree.ElementTree as ET

import pandas as pd
from sklearn.model_selection import train_test_split
import cv2

from conf import *


def prepare_csv(ann_path: str, img_path:str):
    files = os.listdir(ann_path)

    boxes_dict = {
        'class': [],
        'fname': [],
        'xmin': [],
        'xmax': [],
        'ymin': [],
        'ymax': [],
        'imheight': [],
        'imwidth': []
    }

    for f in files:
        root = ET.parse(os.path.join(ann_path, f)).getroot()

        for obj in root.iter('object'):
            if obj.find('name').text not in CLASS_LABELS:
                continue

            boxes_dict['class'] += [obj.find('name').text]
            boxes_dict['fname'] += [os.path.join(img_path, f.replace('.xml', '.jpg'))]
            boxes_dict['imheight'] += [cv2.imread(os.path.join(img_path, f.replace('.xml', '.jpg'))).shape[0]]
            boxes_dict['imwidth'] += [cv2.imread(os.path.join(img_path, f.replace('.xml', '.jpg'))).shape[1]]
            boxes_dict['ymax'] += [int(obj.find('bndbox/ymax').text)]
            boxes_dict['ymin'] += [int(obj.find('bndbox/ymin').text)]
            boxes_dict['xmax'] += [int(obj.find('bndbox/xmax').text)]
            boxes_dict['xmin'] += [int(obj.find('bndbox/xmin').text)]

    cl_files = list(set(boxes_dict['fname']))
    cl_files_train, cl_files_val = train_test_split(cl_files, test_size=0.1)

    train_dict = defaultdict(lambda: False, zip(cl_files_train, [True] * len(cl_files_train)))

    im_df = pd.DataFrame.from_dict(boxes_dict)
    im_df['is_train'] = im_df.fname.map(train_dict)
    train_df = im_df[im_df.is_train]
    val_df = im_df[~im_df.is_train]
    train_df.to_csv('data/train.csv')
    val_df.to_csv('data/val.csv')


if __name__ == '__main__':
    prepare_csv('data/Annotations', 'data/JPEGImages')