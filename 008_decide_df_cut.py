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
import matplotlib.pyplot as plt
from kneed import KneeLocator

# source data paths
dataset = "Nguyen2007"
source_path = 'GraphRank/data/processed_'+dataset+'/candidate_df_dict/'+dataset+'_single_words_df.csv'
save_path = 'GraphRank/data/processed_'+dataset+'/candidate_df_dict/'

df_dict = {}
x2 = []

with open(source_path, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for i, row in enumerate(spamreader):
        # print(row)  # ['subject descriptor', '242']
        k, v = row[0], float(row[1])
        if v < 5:
            continue
        df_dict[i] = v
        x2.append(v)



# Plot the elbow
x1 = list(range(len(x2)))
plt.figure(figsize=(8, 5))
plt.cla()
plt.plot(x1, x2)  # , 'bx-')

kn = KneeLocator(x1, x2, S=4, curve='convex', direction='decreasing')  # 20220510: 10=34, 20=18
print('elbow point (x1)', kn.knee, 'value (x2)', df_dict[kn.knee])  # k value
print('elbow point (x1)', kn.knee, 'value (x2)', x2[kn.knee])  # k value

plt.vlines(kn.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
# plt.ylim(-0.05,3.65)
# plt.yticks([item/5 for item in list(range(0,18,1))])
# x_tick = list(range(0, 360, 10))
# plt.xticks(x_tick, rotation=90)
plt.xlabel('k')  # 62 # best k is c's value, if c starts at 301, k will be 362
plt.ylabel('DF')
plt.title('The Elbow')  # k=42(20220126)

plt.tight_layout()
plt.show()
# plt.savefig(save_path +'elbow_shap_score_'+str(kn.knee)+'.png', dpi=300)

