# importing modules
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
import json
from small_tools import get_files, read_text, csv_writer

ps = PorterStemmer()

project_path = 'GraphRank/'
dataset = 'Nguyen2007'
doc_path = 'GraphRank/data/'+dataset+'/processed_docsutf8/'
key_path = 'GraphRank/data/'+dataset+ '/keys/'

save_path =  'GraphRank/data/processed_'+dataset+'/stem_keys.csv'

w = csv_writer(save_path)
files = get_files(doc_path)
for c, file in enumerate(files[:]):
    # print('file:', file)

    document_id = file.replace('.txt', '')
    # text = read_text(doc_path+file)

    keys = read_text(key_path+file.replace('txt', 'key'))
    keys = keys.split('\n')

    # other json format data are all saved stemmed keys
    stemmed_keys = []
    for key in keys:
        sub_tokens = word_tokenize(key)  # get tokens in a sentence
        sub_tokens_stem = [ps.stem(w) for w in sub_tokens]  # get stem of tokens in a sentence
        stemmed_key = ' '.join(sub_tokens_stem)
        if stemmed_key:
            stemmed_keys.append(stemmed_key)
    # print('stemmed_keys:', stemmed_keys)

    w.writerow([document_id] + stemmed_keys)


