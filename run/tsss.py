#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2019-09-21 20:21
@Author  : Wang Xin
@Email   : wangxin_buaa@163.com
@File    : tsss.py
"""


import numpy as np


qvecs = np.zeros((16, 10))
np.random.seed(1)
for i in range(10):
    qvecs[:, i] = np.random.random(16)

dbvecs = np.zeros((16, 20))

np.random.seed(11)
for i in range(15):
    dbvecs[:, i] = np.random.random(16)

qvecs = np.array(qvecs)
dbvecs = np.array(dbvecs)


scores = np.dot(dbvecs.T, qvecs)     # score越大表示越相似
print(scores)
ranks = np.argsort(-scores, axis=0)  # 从小到大排序
print(ranks)
print(ranks[:, 0].shape)


rt_list = []

for i in range(10):
    rt_list.append(list(ranks[0:10, i]))

print(rt_list)
