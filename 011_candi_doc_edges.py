import csv
import sys
csv.field_size_limit(sys.maxsize)

import os
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from small_tools import make_dir

project_path = 'GraphRank/'
dataset = 'Nguyen2007'

doc_path = project_path + 'data/processed_'+dataset+'/candi_emb_stemmed/'
topic_emb_path = project_path + 'data/processed_'+dataset+'/doc_emb/'
save_path = project_path + 'data/processed_'+dataset+'/candi_doc_edges/'
make_dir(save_path)

files = os.listdir(doc_path)
for i, file in enumerate(files):
    files[i] = file[:]

files = files[:]


def str2float(_input_str, spliter=", "):

    _input_str = _input_str.replace("), array", "").replace("array", "")

    for _i in range(10):
        if _input_str[-1] in [' ', ')', "]"]:
            _input_str = _input_str[:-1]
        if _input_str[0] in [' ', "(", "["]:
            _input_str = _input_str[1:]

    # print(_input_str)

    _input_str = _input_str.replace('\n', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').split(spliter)
    _out_array = np.array([float(_item.replace('[', '').replace(']', '').replace('(', '').replace('(', '').replace('dtype=float32', '')) for _item in _input_str])
    return _out_array


for n, file in enumerate(files):
    print(n, file)  # 157 edge_60_emb.csv

    # get topic emb
    with open(topic_emb_path + file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')

        topic_embs = []
        for row in spamreader:
            k, v = row[0], str2float(row[1], spliter=' ')
            # print(k, len(v))
            topic_embs.append(v)
        topic_emb = np.sum(topic_embs, 0)

    # print(len(topic_emb))  # 768

    # get candi emb
    with open(doc_path + file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')

        for row in spamreader:
            # print(k, len(row[1].split("'], [")))
            # if len(row[1].split("'], [")) != 2:  # sometimes, data use not single quote mark but double quote mark
                # print(k, row[1])
            k = row[0]
            if not k:
                continue
            if len(row[1].split("'], [")) != 2:
                embs = row[1].split('"], [')[1]  # remove word strings
            else:
                embs = row[1].split("'], [")[1]  # remove word strings
            embs = embs.replace('\n', '').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ')
            embs = embs.split(", dtype=float32")[:-1]  # split into arrays
            # print('---', k)
            embs = [str2float(item) for item in embs]
            # print(len(embs))
            candi_emb = np.sum(embs, 0)
            # print(len(candi_emb))  # 768

            cos_simi = cosine_similarity([topic_emb], [candi_emb])
            # print(cos_simi)

            w1 = csv.writer(open(save_path + file.replace('_emb.csv', '_topic.csv'), "a"))
            w1.writerow([[k, "$$$$$$"], cos_simi[0][0]])
    print(n, file)  # edge_J-3.csv_emb.csv
