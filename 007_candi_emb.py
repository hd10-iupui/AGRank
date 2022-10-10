import csv
import os
import time
import numpy as np
import mxnet as mx
from bert_embedding import BertEmbedding

ctx = mx.gpu(0)
bert = BertEmbedding(ctx=ctx, max_seq_length=64, batch_size=8)


start_time = time.time()

dataset = 'Nguyen2007'
project_path = 'GraphRank/'
doc_path = 'GraphRank/data/processed_'+dataset+'/graph_edges_stemmed/'

files = os.listdir(doc_path)

files = files[:]


save_path = 'GraphRank/data/processed_'+dataset+'candi_emb_stemmed/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

global_emb_dict = {}

for n, file in enumerate(files):
    print(n + 1, "th file", file)  # edge_FT941-1750.csv


    local_emb_dict = {}

    with open(doc_path + file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')

        for row in spamreader:
            # print(row)
            node1, node2 = row[0], row[1]
            for node in [node1, node2]:
                # if node in local_emb_dict.keys():
                    # """node is candidate,"""
                    # continue
                if node in global_emb_dict.keys():
                    node_emb = global_emb_dict[node]
                else:
                    node_emb = bert([node])
                local_emb_dict[node] = node_emb
                global_emb_dict[node] = node_emb

    w2 = csv.writer(open(save_path + file.replace('.csv', '_emb.csv'), "a"))
    for k, v in local_emb_dict.items():
        w2.writerow([k, v])

    crt_time = time.time()
    print(n + 1, "th file", file, "running time", crt_time - start_time)




