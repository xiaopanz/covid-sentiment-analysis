import numpy as np
import torch
from torch import nn
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
import matplotlib.pyplot as plt
import pickle
from sklearn.naive_bayes import MultinomialNB
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from collections import Counter
import json
import transformers as ppb
nltk.download('stopwords')

# For DistilBERT:
model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')

# Load pretrained model/tokenizer
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)

TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
stop_words = stopwords.words("english")
stemmer = SnowballStemmer("english")

class RNN(nn.Module):
    def __init__(self,
                 embedding_dim = 768, 
                 hidden_dim = 256,
                 output_dim = 1,
                 n_layers = 2,
                 bidirectional = True,
                 dropout = 0.25):
        
        super().__init__()
                
        
        self.rnn = nn.GRU(embedding_dim,
                          hidden_dim,
                          num_layers = n_layers,
                          bidirectional = bidirectional,
                          batch_first = True,
                          dropout = 0 if n_layers < 2 else dropout)
        
        self.out = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, embedded):
        
        _, hidden = self.rnn(embedded)
                
        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
                        
        output = self.out(hidden)
                
        return output

def preprocess(text, stem=False):
    # Remove link,user and special characters
    text = re.sub(TEXT_CLEANING_RE, ' ', str(text).lower()).strip()
    #Convert www.* or https?://* to URL
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',text)
    
    #Convert @username to __USERHANDLE
    text = re.sub('@[^\s]+','__USERHANDLE',text)  
    
    #Replace #word with word
    text = re.sub(r'#([^\s]+)', r'\1', text)
    
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

def gen_features(data):
    tokenized = []
    labels = []
    for tweet in data:
        text = preprocess(tweet['text'])
        labels.append(tweet['label'] / 4)
        tokenized.append(tokenizer.encode(text, add_special_tokens=True))
    max_len = 72
    padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized])
    attention_mask = np.where(padded != 0, 1, 0)
    input_ids = torch.tensor(padded)  
    attention_mask = torch.tensor(attention_mask)

    with torch.no_grad():
        last_hidden_states = model(input_ids, attention_mask=attention_mask)

    features = last_hidden_states[0][:,0,:].numpy()

    return features, labels

def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    y = torch.FloatTensor(y)
    correct = torch.eq(rounded_preds, y) #convert into float for division 
    acc = correct.sum() / len(correct)
    return acc

def test_lr(features, labels):
    lr_clf = pickle.load(open('model/best_lr', 'rb'))
    print(lr_clf.score(features, labels))

def test_rnn(features, labels):
    device = torch.device('cpu')
    model = RNN()
    model.load_state_dict(torch.load('model/best_RNN.pt', map_location=device))
    preds = []
    for feature in features:
        feature = np.expand_dims(feature, axis=0)
        feature = torch.from_numpy(feature)
        feature = feature.unsqueeze(1)
        pred = model(feature).squeeze(1)
        preds.append(pred)
    preds = torch.FloatTensor(preds)
    print(binary_accuracy(preds, labels))

def test(path):
    with open(path) as f:
        test_data = json.loads(f.read())
    features, labels = gen_features(test_data)
    print(features.shape)
    test_lr(features, labels)
    test_rnn(features, labels)

def main():
    test('twitter-covid-data/labeled/full-labeled.json')

if __name__ == '__main__':
    main()


