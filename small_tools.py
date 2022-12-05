"""this collection of code include every general tools we use daily so that new definition can save a lot of time."""
import os
import csv
import sys
csv.field_size_limit(sys.maxsize)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random


def get_files(path):
    files = os.listdir(path)
    return files


def read_csv(file):
    rows = []
    with open(file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            rows.append(row)
    return rows


def remove_space(text):
    # print('a'+text+'b')
    while text[0] == ' ':
        text = text[1:]
    while text[-1] == ' ' or text[-1] == '\n':
        text = text[:-1]
    return text


def load_keys(my_key):
    keys = []
    with open(my_key, "r") as f:
        for line in f:
            # print([line])
            if not line.replace(' ', '') or not line.replace('\n', ''):
                continue
            k = remove_space(line)
            keys.append(k.lower())
    return keys


def f1(a, b):
    return a * b * 2 / (a + b)


def read_text(path):
    output = ''
    f = open(path, encoding='utf-8')
    for line in f:
        output += line
    f.close()
    return output


def clean_punc_number(sentences):
    # remove punctuations, numbers and special characters
    clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")
    return clean_sentences


def remove_stopwords(input):
    stopwords = []
    my_file = '/home/resadmin/haoran/IJCAI/UGIR_stopwords.txt'
    with open(my_file, "r") as f:
        for line in f:
            if line:
                stopwords.append(line.replace('\n', ''))
    # 319, zhuyaoshi daici, guanci, lianci, jieci
    input = input.split()
    sen_new = " ".join([i for i in input if i not in stopwords])
    return sen_new


def csv_writer(path, write_type='a'):
    w = csv.writer(open(path,  write_type))
    return w


def remove_punc(input_text):
    sentences = [input_text]
    clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")
    return clean_sentences[0]


def quick_plot(y_list, x_list=None, show=True):
    if not x_list:
        x_list = [int(item+1) for item in list(range(len(y_list)))]
    plt.plot(x_list, y_list)
    if show:
        plt.show()
    else:
        plt.savefig()

# above before 0403, below after 0403


def make_dir(input_path):
    if not os.path.exists(input_path):
        os.makedirs(input_path)


# above before 0412, below after 0412

def cut_off_percent(input_list, percent=25):
    cut_off = np.percentile(input_list, [percent])[0]
    return cut_off


def clean_head_tail_block(input_str):
    for i in range(99):
        if input_str[0] == ' ':
            input_str = input_str[1:]
        else:
            break
    for j in range(99):
        if input_str[-1] == ' ':
            input_str = input_str[:-1]
        else:
            break

    return input_str


def compare_two_list(list_1, list_2):

    unique_1 = []
    unique_2 = []

    for item in list_1:
        if item not in list_2:
            unique_1.append(item)

    for item in list_2:
        if item not in list_1:
            unique_2.append(item)

    print('unique in list 1:')
    print(unique_1)
    print('-'*20, '\nunique in list 2:')
    print(unique_2)


# ------ 20220706
def merge_dict(x, y):
    # print(x)
    # print(y)
    for k, v in x.items():
        if k in y.keys():
            y[k] = y[k] + v
        else:
            y[k] = v
    # print(y)


def merge_merge_dict(x, y):
    for km, vm in x.items():
        if km in y.keys():
            merge_dict(vm, y[km])
        else:
            y[km] = vm


# 20220728

def random_numbers(want_number, total_number):
    if want_number > total_number:
        print('error')
    # to generate 'want_number' random numbers out of 'total number'
    generated = []
    total_number_list = list(range(total_number))
    while len(generated) < want_number:
        new = random.choice(total_number_list)
        if new in generated:
            a = 1
        else:
            generated.append(new)
    return generated


# 20220810
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

ps = PorterStemmer()


def stem(sentence):
    # sentence = "Programmers program with programming languages"
    words = word_tokenize(sentence)  # if not token, it will just stem the last word
    output = []
    for w in words:
        # print(w, " : ", ps.stem(w))
        output.append(ps.stem(w))
    return ' '.join(output)


def inverse_the_dict(input_dict):
    output_dict = {}
    for k, v in input_dict.items():
        if v in output_dict.keys():
            output_dict[v] += [k]
        else:
            output_dict[v] = [k]
    return output_dict


