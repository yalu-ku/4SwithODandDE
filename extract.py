import os
import subprocess

path = 'dataset/VOCdevkit/VOC2012/JPEGImages'

for i in os.listdir(path):
    print(subprocess.check_output(f'python main.py --source {path}/{i} --od yolov8x --save', shell=True).decode())