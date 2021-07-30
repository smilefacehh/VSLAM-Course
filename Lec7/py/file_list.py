# -*- coding:utf-8 -*-
import os

# with open('../data/images.txt', 'w') as f:
#     files = os.listdir('../data')
#     files.sort()
#     for file in files:
#         f.write(file + '\n')


with open('../data/images.txt', 'w') as f:
    left_files = os.listdir('../data/left')
    right_files = os.listdir('../data/right')
    left_files.sort()
    right_files.sort()
    for l, r in zip(left_files, right_files):
        f.write(l + ' ' + r + '\n')