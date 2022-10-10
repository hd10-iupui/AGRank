from small_tools import get_files, read_csv, load_keys, f1, remove_punc, cut_off_percent
import networkx as nx
import numpy as np
import csv
from string import punctuation
small_punctuations = [item for item in punctuation if item != '-']


project_path = 'GraphRank/'
dataset = 'Nguyen2007'

edge_path_1 = project_path + 'data/processed_' + dataset + '/filtered_graph_edges_stemmed/'

edge_path_2 = project_path + 'data/processed_' + dataset + '/candi_doc_edges/'

edge_path_sent = project_path + 'data/processed_' + dataset + '/candi_sent_edges/'

keys_path =  'GraphRank/data/processed_'+dataset+'/stem_keys.csv'

df_path = 'GraphRank/data/processed_'+dataset+'/candidate_df_dict/' + dataset + '_single_words_df.csv'

sent_pos_path = 'GraphRank/data/processed_'+dataset+'/sent_position_by_doc/'

# load keys:
all_keys = read_csv(keys_path)
all_keys_dict = dict(zip([item[0] for item in all_keys], [item[1:] for item in all_keys]))

edge_source_1 = get_files(edge_path_1)
edge_source_2 = get_files(edge_path_2)

# load candi_df_dict
candi_df_dict = {}
with open(df_path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
                candi_df_dict[row[0]] = int(row[1])
# print(candi_df_dict)


# setting of structures
doc_emb_in_use = 1
sent_emb_in_use = 0
df_in_use = 1
sent_pos_in_use = 1

# compare the results
f1_dict = {}


problems = []
problem2 = []

for doc_cut in range(1):

    # key parameters
    df = 45
    sent_emb_boost_coef = 1
    sent_cut = 75
    topic_boost = 362
    f1_top = 15
    doc_cut = doc_cut+0

    for ratio in [0.85]:  # 0.85 is the best
        ratio = ratio  # we can see the ratio(alpha) the more large the more time cost  # large than 0.85 is not OK
        # collect filtered edges # example pageRank input [('h','2',0.125),('h','3',0.75),('2','4',1.2),('3','4', 0.375)]
        p_list = []
        r_list = []

        for n, file in enumerate(edge_source_1[:]):  #enumerate(['edge_368.abstr.csv_edges.csv']):  #
            print('file:', n, file)  # file: 0 edge_170_edges.csv

            file_number = file.replace('edge_', '').replace('_edges.csv', '')  # file: --> AP880705-0018

            # load sent pos
            sent_pos_dict = {}
            sent_num = 0
            # file: 0 edge_AP880705-0018_edges.csv  --> AP830325-0143_sent_pos.csv
            with open(sent_pos_path + file[5:].replace('_edges', '_sent_pos'), newline='') as csvfile:
                spamreader = csv.reader(csvfile, delimiter=',')
                for row in spamreader:
                    k = row[0]  # scalabl grid servic discoveri base on uddi * * author are list in alphabet order .,2.0

                    new_weight = (90 - sent_num) / 10  # float(row[1])#
                    v = new_weight  # float(row[1])
                    if new_weight < 1:
                        break
                    if sent_pos_in_use == 1:
                        sent_pos_dict[k.lower()] = v * v
                    else:
                        sent_pos_dict[k.lower()] = 1
                    sent_num += 1

            # print(sent_pos_dict)

            # load candi edges
            local_edge_list = []
            total_candi = []
            for row in read_csv(edge_path_1+file):
                nodes = row[0][2:-2].split("', '")
                node1, node2 = nodes[0], nodes[1]  # "('uddi', 'alphabet order')",0.5532482266426086

                # print(node1,'#', node2)

                # calculate edge's sent pos boost
                pos_boost = 1
                for k in sent_pos_dict.keys():

                    if k.find(node1) != -1 and k.find(node2) != -1:  # using 'and' means we make sure this edge comes from this sent
                        if sent_pos_dict[k] > pos_boost:
                            pos_boost = sent_pos_dict[k]

                # add boost
                weight = float(row[1]) * pos_boost * 2 # multiple 2 or not, may influence other parameters values, but will not influence model performance
                # print(node1, node2, weight)

                # convert our graph to the pageRank friendly feeding data
                local_edge_list.append((node1, node2, weight))
                for node_x in [node1, node2]:
                    if node_x in total_candi:
                        continue
                    else:
                        total_candi.append(node_x)

            # collect candi-topic edges

            # get cutoff thresh
            doc_emb_simi_values = []
            # file: 0 edge_AP880705-0018_edges.csv  --> AP830325-0143_sent_pos.csv
            for row in read_csv(edge_path_2 + file.replace('_edges.csv', '_topic.csv')):
                weight = float(row[1])
                doc_emb_simi_values.append(weight)
            doc_cut_off = cut_off_percent(doc_emb_simi_values, doc_cut)

            for row in read_csv(edge_path_2+file.replace('_edges.csv', '_topic.csv')):
                nodes = row[0][2:-2].split("', '")
                try:
                    node1, node2 = nodes[0], nodes[1]
                except:
                    problems.append([file, row, nodes])
                    continue

                weight = float(row[1])
                if weight < doc_cut_off:
                    continue
                if node1 in total_candi:
                    weight = weight * topic_boost
                    if doc_emb_in_use == 1:
                        local_edge_list.append((node1, node2, weight))

            # load candi-sent edges
            # get cutoff thresh
            sent_emb_simi_values = []
            for row in read_csv(edge_path_sent + file.replace('_edges.csv', '_topic.csv')):
                weight = float(row[1])
                sent_emb_simi_values.append(weight)
            cut_off = cut_off_percent(sent_emb_simi_values, sent_cut)

            for row in read_csv(edge_path_sent + file.replace('_edges.csv', '_topic.csv')):
                nodes = row[0][2:-2].split("', '")
                try:
                    node1, node2 = nodes[0], nodes[1]
                except:
                    problem2.append([file, row, nodes])
                    continue

                # add pos info for sent-emb-bridges
                if node2.find('_0') != -1 or node2.find('_1') != -1:
                    extra_sent_pos_boost = 1
                else:
                    extra_sent_pos_boost = 1

                weight = float(row[1])
                if weight < cut_off:
                    continue
                if node1 in total_candi:
                    weight = weight * sent_emb_boost_coef
                    if sent_emb_in_use == 1:
                        local_edge_list.append((node1, node2, weight))

            # load keys:
            keys = all_keys_dict[file_number]
            # print(keys)
            keys = list(set(keys))

            # build graph
            FG = nx.Graph()
            FG.add_weighted_edges_from(local_edge_list)

            # run pageRank and get result (true f1 score)
            pr = nx.pagerank(FG, ratio)  # the second para is alpha

            # output the prediction rank

            score = 0
            rank = 0
            predictions = []
            # w20 = csv.writer(open(
                # root_path + 'data_0301/prediction_rank_0329/' + file[5:].replace('.abstr.csv_edges', '_prediction_0329'), "a"))  # 'edge_368.abstr.csv_edges.csv'
            for k, v in sorted(pr.items(), key=lambda item: item[1], reverse=True):
                # print(k)
                if not k:
                    continue
                # print(local_edge_list)
                if k.find('_') != -1:  # if a node has no _, it returns -1, and delete last digit
                    k = k[:k.find('_')]
                if k[0] == '$' or k == 'sent':  # drop the sent-node and doc-node during final rank
                    continue
                if df_in_use == 1:
                    if candi_df_dict[k] >= df:
                        continue

                found_punc = 0
                for item in small_punctuations: # drop any candi having punc except '-'
                    if k.find(item) != -1:
                        found_punc = 1
                if found_punc != 0:
                    continue
                rank += 1
                # w20.writerow([k, rank])
                predictions.append(k)  # in case the prediction is less than 10, we calculate acc use min(len(pred), 10)

                # scoring
                if k in keys:
                    tail = '***'
                    if rank <= f1_top:
                        score += 1
                else:
                    tail = ''

                # if rank <= max(f1_top, 20):
                    # print(rank, k, v, tail)
                # if rank == 10:
                    # print('-'*10)

            p = score/(min(f1_top, len(predictions)))  # 31.05 (if fix 10) vs 31.3 (at alpha=0.4) -- not a big problem
            r = score/len(keys)
            p_list.append(p)
            r_list.append(r)
            # print('keys:', keys)
            # break

        p_avg = np.mean(p_list)
        r_avg = np.mean(r_list)
        f1_all = f1(p_avg, r_avg)
        # print('with sent rank boost')
        # print('topic boost:', topic_boost, 'P:', p_avg, 'R:', r_avg, 'F1:', f1_all)
        # print(p_list)
        f1_dict[doc_cut] = f1_all
        print('running doc_cut =', doc_cut, 'f1@'+str(f1_top)+' =', f1_all)#, 'P:', p_avg, 'R:', r_avg)

print("_"*50)
print('doc_cut ', 'f1@'+str(f1_top))
for k, v in f1_dict.items():
    print(k, v)

# print('problems', problems)
# print('problem2', problem2)

