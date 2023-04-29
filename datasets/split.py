import os
import shutil
from collections import defaultdict
from random import sample

txts = os.listdir('texts')
D = defaultdict(int)
indice = sample(range(len(txts)), len(txts)//10)

for idx in indice:
    with open(f'texts/{txts[idx]}', 'r') as f:
        T = f.readlines()
    clsses = [int(t.split()[0]) for t in T]
    for c in clsses:
        D[c] += 1

for i,v in sorted(D.items()):
    print(i, v)
    
os.makedirs('yolo')
os.makedirs('yolo/images/val', exist_ok=True)
os.makedirs('yolo/labels/val', exist_ok=True)
os.makedirs('yolo/images/train', exist_ok=True)
os.makedirs('yolo/labels/train', exist_ok=True)

for idx, txt in enumerate(txts):
    name = txt.split('.')[0]
    dst = 'val' if idx in indice else 'train'
    shutil.copy(f'texts/{name}.txt', f'yolo/labels/{dst}/{name}.txt')
    shutil.copy(f'VOC2012/JPEGImages/{name}.jpg', f'yolo/images/{dst}/{name}.jpg')
