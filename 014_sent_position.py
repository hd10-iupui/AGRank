import csv
import sys
import os
csv.field_size_limit(sys.maxsize)
from string import punctuation
punctuations = [item for item in punctuation]
import matplotlib.pyplot as plt
import numpy as np
from small_tools import make_dir
from nltk.tokenize import word_tokenize
from nltk.stem import *
from nltk.stem.porter import *  #  Unit tests for the Porter stemmer
from small_tools import read_text, remove_stopwords, get_files, csv_writer, make_dir

# generate sentence position in original

# get files:
project_path = 'GraphRank/'
dataset = 'Nguyen2007'
doc_path = 'GraphRank/data/'+dataset+'/processed_docsutf8/'

save_path_0 =  'GraphRank/data/processed_'+dataset+'/sent_position_by_doc/'
make_dir(save_path_0)

files = get_files(doc_path)
files = files[:]

for f, file in enumerate(files):
    print(f, file)
    # load text
    input_text = read_text(doc_path + file)
    sentences = input_text.split('$$$$$$')

    sent_num = len(sentences)

    if sent_num > 1:
        gradient = 1 / (sent_num - 1)
        gradient_list = [2 - gradient * g for g in range(sent_num)]
    else:
        gradient_list = [1]

    w2 = csv_writer(save_path_0+file.replace('.txt', '_sent_pos.csv'))
    for i in range(sent_num):
        # print(sentences[i], gradient_list[i])
        w2.writerow([sentences[i], gradient_list[i]])



# stem the sentence words and save

# Create a new Porter stemmer.
stemmer = PorterStemmer()

sent_pos_path = save_path_0

save_path_1 = save_path_0.replace('sent_position_by_doc', 'sent_position_by_doc_stemmed')
make_dir(save_path_1)

# load edge weights
files = os.listdir(sent_pos_path)
files = files[:]

for n, file in enumerate(files):
    print(n, file)  # C-71_sent_pos.csv

    w1 = csv.writer(open(save_path_1 + file, "a"))

    with open(sent_pos_path + file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            sent, v = row[0], float(row[1])  # sent, pos_weight

            words1 = word_tokenize(sent)  # print(keys[0][0])  # [['qualitative bond graph', '1'],
            # print(words1)  # tokenizer do not break short-line, so the short line in keys also keep the same
            singles1 = [stemmer.stem(plural) for plural in words1]
            stem_node1 = ' '.join(singles1)

            w1.writerow([stem_node1, v])

    print(n, file)