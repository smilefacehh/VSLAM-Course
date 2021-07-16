# -*- coding:utf-8 -*-
import os

with open('../data/images.txt', 'w') as f:
    files = os.listdir('../data')
    files.sort()
    for file in files:
        f.write(file + '\n')