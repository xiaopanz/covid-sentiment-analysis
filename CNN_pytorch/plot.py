from train_common import *
import utils
import random
from utils import config
from torch.utils.data import DataLoader
import json
import bcolz
import pickle
from model import CNN
import re
import nltk
from nltk import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import gc
import numpy as np
from nltk import word_tokenize
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
def preprocess(text, stem=False):
    TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
    stop_words = stopwords.words("english")
    stemmer = SnowballStemmer("english")
    # Remove link,user and special characters
    text = re.sub(TEXT_CLEANING_RE, ' ', str(text).lower()).strip()
    #Convert www.* or https?://* to URL
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',text)
    
    #Convert @username to __USERHANDLE
    text = re.sub('@[^\s]+','__USERHANDLE',text)  
    
    #Replace #word with word
    text = re.sub(r'#([^\s]+)', r'\1', text)
    text = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", text)
    text = re.sub(r"\'s", " \'s", text)
    text = re.sub(r"\'ve", " \'ve", text)
    text = re.sub(r"n\'t", " n\'t", text)
    text = re.sub(r"\'re", " \'re", text)
    text = re.sub(r"\'d", " \'d", text)
    text = re.sub(r"\'ll", " \'ll", text)
    text = re.sub(r",", " , ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\(", " \( ", text)
    text = re.sub(r"\)", " \) ", text)
    text = re.sub(r"\?", " \? ", text)
    text = re.sub(r"\s{2,}", " ", text)
    #trim
    text = text.strip('\'"')
    
    # Repeating words like hellloooo
    repeat_char = re.compile(r"(.)\1{1,}", re.IGNORECASE)
    text = repeat_char.sub(r"\1\1", text)
    
    tokens = []
    for token in text.split():
        if token not in stop_words:
            if stem:
                tokens.append(stemmer.stem(token))
            else:
                tokens.append(token)
    return " ".join(tokens)

input_file=open('twitter-covid-data/full/full-unlabeled.json', 'r')
json_decode=json.load(input_file)

dict_test = {}
for item in json_decode:
    if item.get('created_at') in dict_test.keys():
        dict_test[item.get('created_at')].append(preprocess(item.get('text')))
    else:
        dict_test[item.get('created_at')] = [preprocess(item.get('text'))]



vectors = bcolz.open(f'glove/6B.50.dat')[:]
words = pickle.load(open(f'glove/6B.50_words.pkl', 'rb'))
word2idx = pickle.load(open(f'glove/6B.50_idx.pkl', 'rb'))
glove = {w: vectors[word2idx[w]] for w in words}
max_sent = 42
random_mat = np.random.normal(scale=0.6, size=(50))

for date, texts in dict_test.items():
    for i in range(len(texts)):
        words = word_tokenize(texts[i])
        emb_sent = np.zeros((max_sent, 50))
        for j in range(min(max_sent,len(words))):
            try:
                emb_sent[j] = glove[words[j]]
            except KeyError:
                emb_sent[j] = random_mat
        dict_test[date][i] = emb_sent
del glove
gc.collect()

model = CNN()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Attempts to restore the latest checkpoint if exists
model, start_epoch, stats = restore_checkpoint(model, config('cnn.checkpoint'))

pos_rates = {}
for date, texts in dict_test.items():
    pos = 0
    te_loader = DataLoader(texts, batch_size=len(texts), shuffle=False)
    for X in te_loader:
        with torch.no_grad():
            X = X.float()
            output = model(X)
            predicted = predictions(output.data)
            for res in predicted:
                if res == 1:
                    pos += 1
        pos_rates[date] = [pos, len(predicted)]
        
date_list = []
rate_list = []
for month in range(2, 12):
    pos, total = 0, 0
    for day in range(1,10):
        #"2020-02-16"
        if month < 10:
            if day != 10:
                date = f'2020-0{month}-0{day}'
            else:
                date = f'2020-0{month}-{day}'
        else:
            if day != 10:
                date = f'2020-{month}-0{day}'
            else:
                date = f'2020-{month}-{day}'
        if date in pos_rates.keys():
            pos += pos_rates[date][0]
            total += pos_rates[date][1]
    if total > 0 :
        dates = f'{month}-1~10'
        print(dates, pos/total*100)
        date_list.append(dates)
        rate_list.append(pos/total*100)
    
    pos, total = 0, 0
    for day in range(11,20):
        if month < 10:
            date = f'2020-0{month}-{day}'
        else:
            date = f'2020-{month}-{day}'
        if date in pos_rates.keys():
            pos += pos_rates[date][0]
            total += pos_rates[date][1]
    if total > 0 :
        dates = f'{month}-11~20'
        print(dates, pos/total*100)
        date_list.append(dates)
        rate_list.append(pos/total*100)

    pos, total = 0, 0
    for day in range(21, 31):
        if month < 10:
            date = f'2020-0{month}-{day}'
        else:
            date = f'2020-{month}-{day}'
        if date in pos_rates.keys():
            pos += pos_rates[date][0]
            total += pos_rates[date][1]
    if total > 0 :
        dates = f'{month}-21~'
        print(dates, pos/total*100)
        date_list.append(dates)
        rate_list.append(pos/total*100)

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

figure(figsize=(24, 10), dpi=80)
plt.plot(date_list, rate_list, color='deepskyblue', linewidth=3, alpha=1)
plt.xlabel('Date (2020)')
plt.ylabel('Percentage of Positive Posts(%)')
plt.savefig('trend.png',)