import csv
import os
import time
import numpy as np
import mxnet as mx
from bert_embedding import BertEmbedding
from small_tools import read_text
from configparser import ConfigParser
from swisscom_ai.research_keyphrase.preprocessing.postagging import PosTaggingCoreNLP
from swisscom_ai.research_keyphrase.model.input_representation import InputTextObj
from swisscom_ai.research_keyphrase.model.extractor import extract_candidates

ctx = mx.gpu(0)
bert = BertEmbedding(ctx=ctx, max_seq_length=768, batch_size=8)


def load_local_corenlp_pos_tagger():
    """Need to start StanfordNLP first, copy below two lines to cmd:
    cd GraphRank/stanford-corenlp-full-2018-02-27/
    java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload tokenize,ssplit,pos -status_port 9000 -port 9000 -timeout 15000 &
    """
    config_parser = ConfigParser()
    config_parser.read('config.ini')  # in the GraphRank/
    host = config_parser.get('STANFORDCORENLPTAGGER', 'host')
    port = config_parser.get('STANFORDCORENLPTAGGER', 'port')
    return PosTaggingCoreNLP(host, port)


ptagger = load_local_corenlp_pos_tagger()


start_time = time.time()

project_path = 'GraphRank/'
dataset = 'Nguyen2007'
text_path = 'GraphRank/data/Nguyen2007/processed_docsutf8/'

files = os.listdir(text_path)

files = files[:]

# save emb generated on raw sent
save_path_1 = project_path + 'data/processed_'+dataset+'/sent_emb/'
if not os.path.exists(save_path_1):
    os.makedirs(save_path_1)


for n, file in enumerate(files):
    print(file)  # AP880926-0203.txt

    candidates = []

    # load sentences
    sentences = read_text(text_path + file).split("$$$$$$")  # C-1.txt
    sentences = [item.lower() for item in sentences]

    # assign candi into sentences
    for s_index, sent in enumerate(sentences):
        # print(sent)

        # emb by raw sent
        node_emb_1 = bert([sent])  # cx_mx_bert take and only take list input
        nodes_1 = node_emb_1[0][0]
        embs_1 = node_emb_1[0][1]

        # save raw sent emb by sentence  # edge_C-1.csv_edges.csv
        w1 = csv.writer(open(save_path_1 + 'emb_' +file.replace('.txt', '_sent_')+str(s_index)+'.csv', "a"))
        for i, k in enumerate(nodes_1):
            # print(k, embs[i])
            w1.writerow([k, embs_1[i]])

    # print process by doc
    crt_time = time.time()
    print(n + 1, "th file", file, "running time", crt_time - start_time)




