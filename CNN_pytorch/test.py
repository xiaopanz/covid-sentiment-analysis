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

input_file=open('twitter-covid-data/labeled/full-labeled.json', 'r')
json_decode=json.load(input_file)
x_test, y_test = [], []
for item in json_decode:
    y_test.append(item.get('label')//4)
    x_test.append(preprocess(item.get('text')))


vectors = bcolz.open(f'glove/6B.50.dat')[:]
words = pickle.load(open(f'glove/6B.50_words.pkl', 'rb'))
word2idx = pickle.load(open(f'glove/6B.50_idx.pkl', 'rb'))
glove = {w: vectors[word2idx[w]] for w in words}
max_sent = 42
random_mat = np.random.normal(scale=0.6, size=(50))

for i in range(len(x_test)):
    words = word_tokenize(x_test[i])
    emb_sent = np.zeros((max_sent, 50))
    for j in range(min(max_sent,len(words))):
        try:
            emb_sent[j] = glove[words[j]]
        except KeyError:
            emb_sent[j] = random_mat
    x_test[i] = emb_sent
del glove
gc.collect()

test_data = []
for i in range(len(x_test)):
    test_data.append([x_test[i], y_test[i]])



model = CNN()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Attempts to restore the latest checkpoint if exists
print('Loading cnn...')
model, start_epoch, stats = restore_checkpoint(model, config('cnn.checkpoint'))


te_loader = DataLoader(test_data, batch_size=128, shuffle=False)
y_true, y_pred = [], []
correct, total = 0, 0
running_loss = []
for X, y in te_loader:
    with torch.no_grad():
        X = X.float()
        output = model(X)
        predicted = predictions(output.data)
        y_true.append(y)
        y_pred.append(predicted)
        total += y.size(0)
        correct += (predicted == y).sum().item()
        running_loss.append(criterion(output, y).item())
te_loss = np.mean(running_loss)
te_acc = correct / total

print(te_acc)