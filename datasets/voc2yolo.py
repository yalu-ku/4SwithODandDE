import os
from xml.etree import ElementTree
from pathlib import Path

CLASS = {
    'aeroplane': 1,
    'bicycle': 2,
    'bird': 3,
    'boat': 4,
    'bottle': 5,
    'bus': 6,
    'car': 7,
    'cat': 8,
    'chair': 9,
    'cow': 10,
    'diningtable': 11,
    'dog': 12,
    'horse': 13,
    'motorbike': 14,
    'person': 15,
    'pottedplant': 16,
    'sheep': 17,
    'sofa': 18,
    'train': 19,
    'tvmonitor': 20,
}

CC = [0] * 21

def convert(w, h, ltrb):
    ltrb = list(map(int,ltrb))
    dw = 1. / w
    dh = 1. / h
    x = (ltrb[0] + ltrb[2]) / 2.0
    y = (ltrb[1] + ltrb[3]) / 2.0
    w = ltrb[2] - ltrb[0]
    h = ltrb[3] - ltrb[1]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return ' '.join(map(str, [x, y, w, h])) + '\n'


box = ['xmin', 'ymin', 'xmax', 'ymax']
root = Path('VOC2012')
annot_prefix = root / 'Annotations'
image_prefix = root / 'JPEGImages'
text_prefix = 'texts'

text_prefix.mkdir(exist_ok=True)

for annot in os.listdir(annot_prefix):
    tree = ElementTree.parse(str(annot_prefix / annot))
    filename = tree.find('filename').text
    f_name = Path(filename).stem

    size = tree.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    objs = tree.findall('object')

    classes = []
    with open(f'{text_prefix / f_name}.txt', 'w') as f:
        for obj in objs:
            cls = CLASS[obj.findtext('name')]
            classes.append(cls)
            bbox = obj.find('bndbox')
            ltrb = list(map(float, [bbox.findtext(i) for i in box]))
            line = convert(width, height, ltrb)
            f.write(f'{cls} {line}')

