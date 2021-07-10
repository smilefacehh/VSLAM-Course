# -*- coding:utf-8 -*-
import os

with open('../data/images.txt', 'w') as f:
    files = os.listdir('../data/images')
    files.sort()
    for file in files:
        f.write(file + '\n')