import csv
import os
import sys
csv.field_size_limit(sys.maxsize)
from string import punctuation
punctuations = [item for item in punctuation]
import matplotlib.pyplot as plt
import numpy as np
from small_tools import make_dir
from nltk.tokenize import word_tokenize
from nltk.stem import *
from nltk.stem.porter import *  #  Unit tests for the Porter stemmer


# Create a new Porter stemmer.
stemmer = PorterStemmer()

# source data paths
dataset = 'Nguyen2007'
root_path = 'GraphRank/'
edge_source_path = 'GraphRank/data/processed_'+dataset+'/graph_edges/'
save_path = 'GraphRank/data/processed_'+dataset+'/graph_edges_stemmed/'
make_dir(save_path)

# load edge weights
files = os.listdir(edge_source_path)
files = files[:]


# open edge file and build un-directional edge list for each file
for n, file in enumerate(files):
    print(n, file)

    w1 = csv.writer(open(save_path + file, "a"))
    with open(edge_source_path + file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            node1, node2, edge_value = row[0], row[1], float(row[2])  # node A; node B, edge_weight
            # print('A', node1, node2, edge_value)

            for punc in punctuations:
                node1 = node1.replace(' ' + punc + ' ', punc)
                node2 = node2.replace(' ' + punc + ' ', punc)
            """ replace '-' or all punc, there is not big difference"""

            # cut head
            # no cut anymore

            # we do not consider self-edge
            if node1 == node2:
                continue

            words1 = word_tokenize(node1)  # print(keys[0][0])  # [['qualitative bond graph', '1'],
            # print(words1)  # tokenizer do not break short-line, so the short line in keys also keep the same
            singles1 = [stemmer.stem(plural) for plural in words1]
            stem_node1 = ' '.join(singles1)

            words2 = word_tokenize(node2)  # print(keys[0][0])  # [['qualitative bond graph', '1'],
            # print(words2)  # tokenizer do not break short-line, so the short line in keys also keep the same
            singles2 = [stemmer.stem(plural) for plural in words2]
            stem_node2 = ' '.join(singles2)

            # print('B', node1, node2, edge_value)

            w1.writerow([stem_node1, stem_node2, edge_value])
