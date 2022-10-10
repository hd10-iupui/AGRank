import os
import time
import csv
import sys
from string import punctuation

csv.field_size_limit(sys.maxsize)
punctuations = [item for item in punctuation]

# source data paths
dataset = "Nguyen2007"

edge_source_path = 'GraphRank/data/processed_'+dataset+'/graph_edges_stemmed/'

save_path = 'GraphRank/data/processed_'+dataset+'/candidate_df_dict/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

# load edge weights
files = os.listdir(edge_source_path)
files = files[:]

# open edge file and build un-directional edge list for each file

df_dict = {}

for n, file in enumerate(files):

    edge_wt_dict = {}
    local_candidates = []
    with open(edge_source_path + file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            node1, node2, edge_value = row[0], row[1], float(row[2])  # node A; node B, edge_weight

            for node in [node1, node2]:
                if node in local_candidates:  # keep the candi only if unique
                    continue
                else:
                    local_candidates += [node]

    # print(local_candidates)

    # get df
    for k in local_candidates:  # local_candi is a unique candi list

        if k in df_dict.keys():
            df_dict[k] += 1
        else:
            df_dict[k] = 1

    print(n, 'th file', file)

# save the df dict
write_yes = True
w1 = csv.writer(open(save_path + dataset + '_single_words_df.csv', "a"))
x = []
for k, v in sorted(df_dict.items(), key=lambda item: item[1], reverse=True):
    if write_yes:
        w1.writerow([k, v])

    if v>10:
        print(k, v)
        x.append(v)

import matplotlib.pyplot as plt

plt.hist(x, density=False, bins=300)  # density=False would make counts
plt.ylabel('count')
plt.xlabel('df')
plt.show()
# plt.savefig(save_path+'candidate_df_dict.png')
