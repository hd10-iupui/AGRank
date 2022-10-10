import time
import numpy as np
import pickle
import os.path
import csv
import sys
csv.field_size_limit(sys.maxsize)
from configparser import ConfigParser
from swisscom_ai.research_keyphrase.preprocessing.postagging import PosTaggingCoreNLP
from swisscom_ai.research_keyphrase.model.input_representation import InputTextObj
from swisscom_ai.research_keyphrase.model.extractor import extract_candidates
from string import punctuation
punctuations = [item for item in punctuation]
from small_tools import make_dir, clean_head_tail_block, merge_dict


def load_pickle(fname):
    with open(fname, "rb") as f:
        return pickle.load(f, encoding="latin1")  # add, encoding="latin1") if using python3 and downloaded data


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


def get_candidate(sentence):
    tagged = ptagger.pos_tag_raw_text(sentence)
    text_obj = InputTextObj(tagged, 'en')
    candidates = extract_candidates(text_obj, repeat=True)  # , gram=3)

    return candidates


def get_candidate_index(candidates, sentence_words):
    # we append candidates as their order as also keep the repeat
    # print()
    # print('get_candidate_index: candidates', len(candidates), candidates)
    # print('get_candidate_index: sentence words', len(sentence_words), [item+'_'+str(sw) for sw, item in enumerate(sentence_words)])
    """example as C-1 (20220501)
    'log_38', '¢_52', 'number_55', 'documents_57', 'corpus_60'
    'log', 'cents', 'number', 'documents', 'corpus'
    one case: stanford NLP results not 100% represent the raw string;
    another case: matrix has item 'cents' but I fail to extract it.
    conclusion: need to add a double safety to skip the missing matrix part"""
    previous_remain_index = []
    candi_delete_missing = []  # drop the missing candi, or the next pairing step will say numbers not match (20220501)
    local_success = False
    remain_index = list(range(len(sentence_words)))
    total_index = []

    for c, candidate in enumerate(candidates):
            candi_delete_missing.append(candidate)  # append the new candi first, them delete
            if len(remain_index) > 0:  # if this step still has index to use, store them into [previous_remain_index]
                    previous_remain_index = remain_index
            else:  # if last loop kill all index, restore the previous remain index from last step
                    remain_index = previous_remain_index
                    candi_delete_missing = candi_delete_missing[:-2]  # because we are dropping the candi before the last one
                    candi_delete_missing.append(candidate)  # not finish, if the last candi fail, we still need to remove it
            candidate_words = candidate.split()
            local_window = len(candidate_words)
            # print('\ncandidate:', candidate, '   window:', local_window, 'remain_index', remain_index)
            index_collect = []
            for d, index in enumerate(remain_index):
                    # print('checking index: ', index, '       remain_index-1', remain_index)  # content
                    # this index is stored index, will not change during remain-index reducing, do not worry (20220501)
                    if sentence_words[index] == candidate_words[0] and len(remain_index)>=local_window:  # content
                            success = 1
                            index_collect.append(index)  # content
                            for ll in range(local_window - 1):
                                    if sentence_words[index + ll+1] == candidate_words[ll+1]:  # content
                                            success += 1
                                            index_collect.append(index + ll+1)  # content
                            if success == local_window:
                                    remain_index = remain_index[local_window:]  # l itself occupy a digit index  # order
                                    total_index.append(index_collect)
                                    # print('success index: ', index_collect, 'remain_index-3', remain_index)
                                    # if success, stop check rest index
                                    local_success = True
                                    break
                            else:
                                    # if fail to match (success != local window), need to clean the index_collect
                                    index_collect = []
                                    remain_index = remain_index[success:]  # order
                                    # print('failure  index : ', index, 'remain_index-4', remain_index)
                                    local_success = False
                    else:
                            remain_index = remain_index[1:]  # order  # recursively drop the first one
                            # print('remove   index: ', index, '       remain_index-2', remain_index)
                            local_success = False
                            continue

            if c == len(candidates)-1 and not local_success:
                # print('last candi local_success=False', candi_delete_missing[-1])
                candi_delete_missing = candi_delete_missing[:-1]
    return candi_delete_missing, total_index  # return the candi list without missing candi


def pair(list_1, list_2):
    if len(list_1) != len(list_2):
        print('list not same length')
        return 'list not same length'
    else:
        out = []
        for i in range(len(list_1)):
            out.append(list_1[i] + '_' + str(list_2[i]))
        return ' '.join(out)


def block_pun(input):
    output = input
    for i in range(len(input)):
        if input[i] in punctuations:
            output = output.replace(input[i], ' '+input[i]+' ')
    return output


def plot_attn(example, heads, sentence_length, current_sentence):
    shrink_2_list = ['-']  # , '=', '+', '<', '>', '|']  # list of characters who need to shrink 2 pointer
    sum_matrix = np.zeros([sentence_length, sentence_length])  # sentence_length = len(data[record]['tokens'])
    pointer = []
    hashtag_leads = []

    # we want to keep the matrix structure, so add 12 head matrix directly
    for ei, (layer, head) in enumerate(heads):
        attn = example["attns"][layer][head]  # [0:sentence_length, 0:sentence_length]
        attn = np.array(attn)
        attn /= attn.sum(axis=-1, keepdims=True)  # normal matrix value
        sum_matrix += attn

    # get hash leading words and their positions >>> what is not leading? #head words and =+-
    attn_sum = sum_matrix.sum(axis=0, keepdims=True)  # np.shape(attn_sum) = (1,sentence length)
    weights_list = attn_sum[0]
    tokens = example["tokens"]  # [0:sentence_length]

    # --------------------------find and shrink token level attn matrix to word (hashtag leads) level matrix--------------------------
    attn_list = []
    for p in range(len(tokens)):
        if p > 0:
            p_cannot_be_0 = p
        else:
            p_cannot_be_0 = 1
        if tokens[p].find("##") == -1:  # and tokens[p] not in punctuations and tokens[p_cannot_be_0 - 1] not in punctuations:
            hashtag_lead = tokens[p]
            hash_weights = [weights_list[p]]
            # print()
            # print(hashtag_lead)

            shorter = 0
            longer = 0
            for i in range(99):
                try:
                # if p + 1 + i + longer < len(tokens):# and p + 1 + 2 * i + shorter < len(tokens):

                    if tokens[p + 1 + i + longer][0] == "#":
                        hashtag_lead += tokens[p + 1 + i].replace("##", "")
                        hash_weights += weights_list[p + 1 + i]
                        shorter = -1
                        longer = 0
                        # print('step 1 check, i =', i, 'p = ', p)
                        # print(tokens[p + 1 + i], p + 1 + i, tokens[p + 1 + i * 2 + shorter], p + 1 + i * 2 + shorter)
                    # here we consider a=b=c this situation, the first = is p+1+2*0, the second = is p+1+2*1
                    else:
                        # print('** stop search')
                        break
                except:
                # else:
                    break
                    # continue
            # print('NEW: ', hashtag_lead)
            pointer.append(p)  # the pointer p records which word can lead
            hashtag_leads.append(hashtag_lead)
            attn_list.append(hash_weights)

    # shrink token level attn matrix to word (hashtag leads) level matrix
    shrink_matrix = sum_matrix.copy()
    '''the key of matrix-shrink is 'step', pointer is moving, as well as step's position'''
    step = 0
    for p, pt in enumerate(pointer):  # the pointers record which words can lead
        if p != len(pointer) - 1:
            if pt != pointer[p + 1] - 1:  # if later neighbor cannot be a hash leader, it needs shrink
                gram_head = pt - step
                gram_tail = pointer[p + 1] - step
                step += gram_tail - gram_head
                shrink_matrix[0:, gram_head] = np.sum(shrink_matrix[0:, gram_head:gram_tail], axis=1)
                shrink_matrix = np.delete(shrink_matrix, list(range(gram_head + 1, gram_tail)), axis=1)

                shrink_matrix[gram_head] = np.sum(shrink_matrix[gram_head:gram_tail], axis=0) / step
                shrink_matrix = np.delete(shrink_matrix, list(range(gram_head + 1, gram_tail)), axis=0)

    # --------------------------generate this sentence's candidates with index--------------------------
    sentence_words = hashtag_leads  # [1:int((len(hashtag_leads)-1)/2)]
    sentence = ' '.join(sentence_words)
    sentence = sentence.replace('[CLS] ','').replace('[SEP] ', '').replace('[CLS]','').replace('[SEP]', '')
    # print()
    # print('1', sentence)
    # print('2', hashtag_leads)
    # print(file)
    # print(sentence)
    raw_candidates = get_candidate(current_sentence)
    # candidates = candidates[1:int((len(candidates) - 1) / 2)] + candidates[int((len(candidates)-1)/2)+1: -1]

    # filter candidate
    #  = ['new', 'different', 'available', 'important', 'possible']
    candidates = []
    for node in raw_candidates:
        # print(node[:node.find(' ')])
        # if node[:node.find(' ')] in cut_list:
            # print()
            # node = node[node.find(' ') + 1:]

        if node in candidates:
            continue
        else:
            candidates += [node]

    candidates += candidates
    # print('3', len(candidates), candidates)
    # candidates = [item.replace('-', ' - ').replace('/', ' / ').replace('  ',' ').replace('.', ' . ').replace('+', ' + ').replace('@', ' @ ') for item in candidates]
    candidates = [item for item in candidates if item not in ['n\'ts', 'p&id']]
    candidates = [item for item in candidates if item[0] != '<' and item[-1] != '>']
    candidates= [item for item in candidates if item.find('@') == -1]
    candidates = [block_pun(item) for item in candidates]
    no_candi = False
    if not candidates:
        no_candi = True

    # print('4', len(candidates), candidates)
    # print(sentence_words)

    candidates, candidate_index = get_candidate_index(candidates, sentence_words)
    if candidate_index or no_candi:  # this line matrix and sentence make some sense, else skip this matrix

        # print('6', len(sentence_words), len(shrink_matrix), len(shrink_matrix[0]))

        # print('7', [word + '_' + str(sentence_words.index(word)) for word in sentence_words])

        indexed_candidates = []
        for c, candidate in enumerate(candidates):
            candidate = clean_head_tail_block(candidate)
            try:
                paired_candidate = pair(candidate.split(), candidate_index[c])
                # print(paired_candidate)
                if paired_candidate != 'list not same length':
                    indexed_candidates.append(paired_candidate)
            except:
                print()
                print(c, candidate)
                print('a', candidate.split())

                print('true candidates', (len(candidates)), candidates)
                print('converted candidate_index', (len(candidate_index)), candidate_index)
                # because in line 315: candidate_index = get_candidate_index(candidates, sentence_words)
                print('sentence_words', len(sentence_words), sentence_words)
                # so the problem happens in def: get_candidate_index()
                # stop
        # indexed_candidates = [pair(candidate.split(), candidate_index[c]) for c, candidate in enumerate(clean_candidates)]
        # print('8', indexed_candidates)
        # 4 ['   <   article   >       <   fm   >       <   atl   >   ', 'compression', '  <   / ip1  >     <  p  >  ', 'good reasons', 'memory requirements', '  <  ss1  >     <  st  >  ', 'method', ' <  / st > ', ' <  / sec > ', '  <    /  bdy  >     <    /  article  >   figure', 'journal article', 'xml', '   <   article   >       <   fm   >       <   atl   >   ', 'compression', '  <   / ip1  >     <  p  >  ', 'good reasons', 'memory requirements', '  <  ss1  >     <  st  >  ', 'method', ' <  / st > ', ' <  / sec > ', '  <    /  bdy  >     <    /  article  >   figure', 'journal article', 'xml']
        # 5 [[1, 2, 3, 4, 5, 6, 7, 8, 9], [11], [100, 101, 102, 103, 104, 105, 106], [292, 293], [314, 315], [323, 324, 325, 326, 327, 328], [332], [334, 335, 336, 337], [341, 342, 343, 344], [348, 349, 350, 351, 352, 353, 354, 355, 356], [360, 361], [364]]
        # IndexError: list index out of range


        # write all edges to csv
        save_path = save_path_0

        w20 = csv.writer(open(save_path + "edge_" + file + '.csv', "a"))

        sentence_edge_dict = {}

        for candidateL in indexed_candidates:
            for candidateR in indexed_candidates:
                # if candidateL != candidateR:
                # print(candidateL, candidateR)

                left_index = [item[item.find('_') + 1:] for item in candidateL.split()]
                right_index = [item[item.find('_') + 1:] for item in candidateR.split()]
                # some candi has many '_'
                word_edge = []
                for wordL in left_index:
                    for wordR in right_index:
                        word_edge.append([wordL, wordR])

                # print(candidateL, candidateR, word_edge)

                attn = 0
                for edge in word_edge:
                    try:  # some candi has many '_'
                        row, col = int(edge[0]), int(edge[1])
                    except:  # like: item[item.find('_') + 1:]
                        row, col = int(edge[0][edge[0].find('_') + 1:]), int(edge[1][edge[1].find('_') + 1:])
                    v = shrink_matrix[row][col]
                    # print(edge[0], edge[1], v)
                    attn += v

                # print(candidateL, candidateR, attn)
                w20.writerow([candidateL, candidateR, attn])

                left_candidate = [item[:item.find('_')] for item in candidateL.split()]
                right_candidate = [item[:item.find('_')] for item in candidateR.split()]
                left_candidate = ' '.join(left_candidate)
                right_candidate = ' '.join(right_candidate)

                # print(left_candidate, right_candidate, attn)
                merge_dict({(left_candidate, right_candidate): attn}, sentence_edge_dict)

                # print()

        # print()
        return sentence_edge_dict

    else:  # if not candidate_index:
        if invest_printing == True:print('this_matrix_failure = True')
        return 'this_matrix_failure = True'



# attn extractor ######################################################


start_time = time.time()

dataset = 'Nguyen2007'
files_path = dataset + '/docsutf8/'
pkl_path = dataset + '/processed_'+dataset+'/sentence_paired_text/'
paired_text_path = pkl_path
bert_name = "orgbert"

save_path_0 = 'GraphRank/data/processed_'+dataset+'/graph_edges_raw/'
make_dir(save_path_0)
save_path_1 = 'GraphRank/data/processed_'+dataset+'/graph_edges/'
make_dir(save_path_1)

files = os.listdir(files_path)
for i, file in enumerate(files):  # files = 243
    files[i] = file[:-4]

files = files[:]

problem_files = {}

invest_printing = False
ptagger = load_local_corenlp_pos_tagger()




for n, file in enumerate(files):

    print('dealing', n, file)
    last_time = time.time()

    # load token-level attn matrix
    attn_extracting_dir = pkl_path + file + "_" + bert_name + '_attn.pkl'  # 2.abstr_orgbert_attn.pkl
    data = load_pickle(attn_extracting_dir)

    # build a total dict of edges
    edges_value_sum = {}

    total_edge_count = 0

    # load text  # 2.abstr_sentence_paired.txt
    all_sentence = []
    with open(paired_text_path + file + '_sentence_paired.txt', "r") as f:
        for line in f:
            line = line.replace('\n' , '')
            if line:
                if line in all_sentence:
                    continue
                else:
                    all_sentence.append(line)

    matching_sent = 0
    health_score = len(data)
    for record in range(len(data)):
        # print('total data number', len(data), 'total sent number', len(all_sentence), 'current data', record, 'current sent', matching_sent)
        if matching_sent == len(all_sentence):
            print('all sentences are used up, current data', record, 'is skipped.')
            health_score = health_score - 1
            continue

        sentence_length = len(data[record]['tokens'])
        this_matrix_failure = False

        # load 12th layer all 12 heads
        layer = 11
        sentence_edge_dict = plot_attn(data[record], [(layer, 0), (layer, 1), (layer, 2), (layer, 3),
                                                      (layer, 4), (layer, 5), (layer, 6), (layer, 7),
                                                      (layer, 8), (layer, 9), (layer, 10),
                                                      (layer, 11)],
                                       sentence_length, all_sentence[matching_sent])

        # print(sentence_edge_dict)
        if sentence_edge_dict == 'this_matrix_failure = True':
            health_score = health_score - 1
            continue

        merge_dict(sentence_edge_dict, edges_value_sum)
        matching_sent+= 1  # is this matrix success, move to next sent (20220501)
        # for k, v in sentence_edge_dict.items():
        # print(k, v)

    # write all edges to csv
    health_score = health_score/len(data)

    w21 = csv.writer(open(save_path_1 + "edge_" + file + '.csv', "a"))

    s = 0

    for k, v in edges_value_sum.items():
        # print(s, k[0], k[1], v)
        w21.writerow([k[0], k[1], v])
        s += 1

    # print(len(edges_value_sum))

    run_time = time.time()
    print(n, "th file", file, "running time", run_time - last_time, run_time - start_time, 'health score', health_score)
    if health_score < 0.9:
        problem_files[file] = health_score

    print('problem_files', len(problem_files), problem_files)

print('problem_files', len(problem_files))
for k, v in problem_files.items():
    print(k, v)
