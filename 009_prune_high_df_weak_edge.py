import csv
import os
import sys
from string import punctuation
import matplotlib.pyplot as plt
import numpy as np
from small_tools import make_dir

csv.field_size_limit(sys.maxsize)
punctuations = [item for item in punctuation]
small_punctuations = [item for item in punctuation if item != '-']

# source data paths
dataset = 'Nguyen2007'
data_path = 'GraphRank/data/processed_' + dataset + '/'
edge_source_path = data_path + 'graph_edges_stemmed/'
save_path = data_path + 'filtered_graph_edges_stemmed/'
make_dir(save_path)

# load edge weights
files = os.listdir(edge_source_path)
files = files[:]

df_cut = 56  # the doc freq thresh got from 008 step-2

# load candi_df_dict
candi_df_dict = {}
with open(data_path + 'candidate_df_dict/'+dataset+'_single_words_df.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
                candi_df_dict[row[0]] = int(row[1])


# open edge file and build un-directional edge list for each file
for n, file in enumerate(files):

    print(n, file)  # 0 edge_FT941-1750.csv

    edge_wt_dict = {}
    with open(edge_source_path + file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            node1, node2, edge_value = row[0], row[1], float(row[2])  # node A; node B, edge_weight

            # we do not consider self-edge
            if node1 == node2:  # (some new self-edge since stemmed; 20220510)
                # print('3', row)
                continue

            # drop high df nodes
            if candi_df_dict[node1] >= df_cut or candi_df_dict[node2] >= df_cut:
                continue

            found_punc = 0
            for item in small_punctuations:
                if node1.find(item) != -1 or node2.find(item) != -1:
                    found_punc = 1
                    break
            if found_punc != 0:
                # print('4', row)
                continue

            # we consider un-directional, node1-node2 is eq to node2-node2
            if (node1, node2) in edge_wt_dict.keys():
                edge_wt_dict[node1, node2] += edge_value  # if same edge, add up weights
            elif (node2, node1) in edge_wt_dict.keys():
                edge_wt_dict[node2, node1] += edge_value  # if opposite edge exists, add to the existed one
            else:
                edge_wt_dict[node1, node2] = edge_value

    local_edge_wts = list(edge_wt_dict.values())
    local_edge_wts.sort()
    # print(local_edge_wts)
    # print(np.percentile(local_edge_wts, [25, 50, 75]))
    cut_off = np.percentile(local_edge_wts, [25])[0]

    w1 = csv.writer(open(save_path + file.replace('.csv', '_edges.csv'), "a"))
    for k, v in edge_wt_dict.items():
        if v >= cut_off:
            w1.writerow([k, v])
