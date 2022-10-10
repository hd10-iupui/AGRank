import csv
import sys
csv.field_size_limit(sys.maxsize)
import os
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from small_tools import make_dir


def str2float(_input_str, spliter=", "):

    _input_str = _input_str.replace("), array", "").replace("array", "")

    for _i in range(10):
        if _input_str[-1] in [' ', ')', "]"]:
            _input_str = _input_str[:-1]
        if _input_str[0] in [' ', "(", "["]:
            _input_str = _input_str[1:]

    _input_str = _input_str.replace('\n', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').split(spliter)
    _out_array = np.array([float(_item.replace('[', '').replace(']', '').replace('(', '').replace('(', '').replace('dtype=float32', '')) for _item in _input_str])
    return _out_array


project_path = 'GraphRank/'
dataset = 'Nguyen2007'

doc_path = project_path + 'data/processed_'+dataset+'/candi_emb_stemmed/'
topic_emb_path = project_path + 'data/processed_'+dataset+'/sent_emb/'
save_path = project_path + 'data/processed_'+dataset+'/candi_sent_edges/'
make_dir(save_path)

# get files names
files = os.listdir(doc_path)
for i, file in enumerate(files):
    files[i] = file[:]

files = files[:]


# get files sentences index
files_1 = os.listdir(topic_emb_path)
file_sent_dict = {}
for i_1, file_1 in enumerate(files_1):
    # print(file_1)  # emb_AP830325-0143_sent_0.csv
    file_1 = file_1.split('_sent_')
    file_name = file_1[0][4:]
    sent_num = file_1[1][:-4]
    if file_name in file_sent_dict.keys():
        file_sent_dict[file_name].append(sent_num)
        file_sent_dict[file_name].sort()  # sort sent index in each file
    else:
        file_sent_dict[file_name] = [sent_num]


# print(file_sent_dict)  # ['0', '1', '2', '3', '4', '5', '6']

# print details when edit and invest the code, hide details when generating data
detail_print = 0
real_saving = 1


for n, file in enumerate(files):
    print('the', n, 'th file,',  file)  # edge_AP830325-0143_emb.csv
    file_num = file[5:].replace('_emb.csv', '')

    # get sent emb
    """for here, exists a situation that one sent has no candi so that the sent-candi-emb file if empty
    so we need dict to map sent-index to sent candi embs"""
    sent_embs = {}
    # load sent index of current  file
    sent_ids = file_sent_dict[file_num]  # ['0', '1', '2', '3', '4', '5'] of file 301
    print('sent_ids', sent_ids) if detail_print == 1 else None

    for sent_id in sent_ids:  # emb_AP830325-0143_sent_0.csv
        """for here, exists a situation that one sent has no candi so that the sent-candi-emb file if empty
        so we need dict to map sent-index to sent candi embs"""
        with open(topic_emb_path + file.replace('_emb', '_sent_'+str(sent_id)).replace('edge_', 'emb_'), newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            token_embs = []
            for row in spamreader:
                k, v = row[0], str2float(row[1], spliter=' ')
                # print(k, len(v))
                token_embs.append(v)
            sent_emb = np.sum(token_embs, 0)

        if token_embs:
            """for here, exists a situation that one sent has no candi so that the sent-candi-emb file if empty
                    so we need dict to map sent-index to sent candi embs; if token embs empty, skip"""
            sent_embs[sent_id] = sent_emb
            print('file', file_num, 'sent', sent_id, 'emb len', len(sent_emb)) if detail_print == 1 else None  # 768
        else:
            print('file', file_num, 'sent', sent_id, 'emb len', 0) if detail_print == 1 else None  # 768

    # get candi emb
    with open(doc_path + file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')

        for row in spamreader:
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
            # print('len(candi_emb)', len(candi_emb))  # 768

            # save candi to sent embs by file is ok
            w1 = csv.writer(open(save_path + file.replace('_emb.csv', '_topic.csv'), "a")) if real_saving == 1 else None

            # calculate cos simi with each sent emb
            for se, sent_emb in sent_embs.items():
                cos_simi = cosine_similarity([sent_emb], [candi_emb])
                print(se, cos_simi) if detail_print == 1 else None
                w1.writerow([[k, "sent_emb_"+str(se)], cos_simi[0][0]]) if real_saving == 1 else None

    print('the', n, 'th file,', file)  # edge_301.abstr.csv_emb.csv
