#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2019-09-21 19:55
@Author  : Wang Xin
@Email   : wangxin_buaa@163.com
@File    : gen_retrieval.py
"""


import numpy as np
import tensorflow as tf


from netvlad_tf.image_descriptor import ImageDescriptor

#

from run.database import AAcchenDB
aachen_db = AAcchenDB(root='/data/vldata/aachen', slice='day_time')

db_imgs, q_imgs = aachen_db.db_imgs, aachen_db.query_imgs

tf.reset_default_graph()
imd = ImageDescriptor(is_grayscale=False)
db_feats = imd.describeAllJpegsInPath(db_imgs, batch_size=1, verbose=True)
q_feats = imd.describeAllJpegsInPath(q_imgs, batch_size=1, verbose=True)

# lll
db_feats = np.array(db_feats).T
q_feats = np.array(q_feats).T

scores = np.dot(db_feats.T, db_feats)
ranks = np.argsort(-scores, axis=0)

match_list = []
# generate match list
for i in range(len(db_imgs)):
    q_list = list(ranks[0:10, i])

    for qi in q_list:
        # match_list.append((db_imgs[i], db_imgs[qi]))
        match_list.append((aachen_db.get_image_name(db_imgs[i]),
                          aachen_db.get_image_name(db_imgs[qi])))


scores = np.dot(db_feats.T, q_feats)
ranks = np.argsort(-scores, axis=0)

for i in range(len(q_imgs)):
    q_list = list(ranks[0:10, i])

    for qi in q_list:
        # match_list.append((q_imgs[i], db_imgs[qi]))
        match_list.append((aachen_db.get_image_name(q_imgs[i]),
                           aachen_db.get_image_name(db_imgs[qi])))

# print(match_list)

match_list_file = open('day_time_match_list_file.txt', 'a')
for q, p in match_list:
    tmp_line = str(q) + ' ' + str(p) + '\n'
    match_list_file.write(tmp_line)

match_list_file.close()

"""
    -------------------------------------------------------------
"""
aachen_db = AAcchenDB(root='/data/vldata/aachen', slice='night_time')

db_imgs, q_imgs = aachen_db.db_imgs, aachen_db.query_imgs

tf.reset_default_graph()
imd = ImageDescriptor(is_grayscale=False)
db_feats = imd.describeAllJpegsInPath(db_imgs, batch_size=1, verbose=True)
q_feats = imd.describeAllJpegsInPath(q_imgs, batch_size=1, verbose=True)

# lll
db_feats = np.array(db_feats).T
q_feats = np.array(q_feats).T

scores = np.dot(db_feats.T, db_feats)
ranks = np.argsort(-scores, axis=0)

match_list = []
# generate match list
for i in range(len(db_imgs)):
    q_list = list(ranks[0:10, i])

    for qi in q_list:
        # match_list.append((db_imgs[i], db_imgs[qi]))
        match_list.append((aachen_db.get_image_name(db_imgs[i]),
                          aachen_db.get_image_name(db_imgs[qi])))


scores = np.dot(db_feats.T, q_feats)
ranks = np.argsort(-scores, axis=0)

for i in range(len(q_imgs)):
    q_list = list(ranks[0:10, i])

    for qi in q_list:
        # match_list.append((q_imgs[i], db_imgs[qi]))
        match_list.append((aachen_db.get_image_name(q_imgs[i]),
                           aachen_db.get_image_name(db_imgs[qi])))

match_list_file = open('night_time_match_list_file.txt', 'a')
for q, p in match_list:
    tmp_line = str(q) + ' ' + str(p) + '\n'
    match_list_file.write(tmp_line)

match_list_file.close()



