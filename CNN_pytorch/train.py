#!/usr/bin/env python
import torch
import numpy as np
import random
from model import CNN
from train_common import *
from utils import config
import utils
import gc
from sklearn.utils import shuffle
import bcolz
import pickle
from torch.utils.data import DataLoader
import pandas as pd
import torch
from matplotlib import pyplot as plt
import re
import nltk
from nltk import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

def _train_epoch(data_loader, model, criterion, optimizer):
    """
    Train the `model` for one epoch of data from `data_loader`
    Use `optimizer` to optimize the specified `criterion`
    """
    for i, (X, y) in enumerate(data_loader):
        # clear parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        X = X.float()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

def _evaluate_epoch(axes, tr_loader, val_loader, model, criterion, epoch,
    stats):
    """
    Evaluates the `model` on the train and validation set.
    """
    y_true, y_pred = [], []
    correct, total = 0, 0
    running_loss = []
    for X, y in tr_loader:
        with torch.no_grad():
            X = X.float()
            output = model(X)
            predicted = predictions(output.data)
            y_true.append(y)
            y_pred.append(predicted)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            running_loss.append(criterion(output, y).item())

    train_loss = np.mean(running_loss)
    train_acc = correct / total
    y_true, y_pred = [], []
    correct, total = 0, 0
    running_loss = []
    for X, y in val_loader:
        with torch.no_grad():
            X = X.float()
            output = model(X)
            predicted = predictions(output.data)
            y_true.append(y)
            y_pred.append(predicted)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            running_loss.append(criterion(output, y).item())
    val_loss = np.mean(running_loss)
    val_acc = correct / total
    stats.append([val_acc, val_loss, train_acc, train_loss])
    utils.log_cnn_training(epoch, stats)
    utils.update_cnn_training_plot(axes, epoch, stats)
    
# def _evaluate_clf(axes, val_loader, model, criterion, epoch, stats):
#     with torch.no_grad():
#         correct = [0,0]
#         total = [0,0]
#         for X, y in val_loader:
#             X = X.float()
#             output = model(X)
#             predicted = predictions(output.data)
#             for i in range(y.size(0)):
#                 if (y[i] == predicted[i]):
#                     correct[y[i]] += 1
#                 total[y[i]] += 1
#         print(correct[0]/total[0], correct[1]/total[1])
        
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

def preprocess_data(X, y):
    # Data Preparation
    # ==================================================

    # Split train/test set
    # TODO: should use cross-validation
    dev_sample_index = -1 * int(0.1 * float(len(y)))

    x_train, x_dev = X[:dev_sample_index], X[dev_sample_index:]
    y_train, y_dev = y[:dev_sample_index], y[dev_sample_index:]
    del X, y
    gc.collect()
    vectors = bcolz.open(f'glove/6B.50.dat')[:]
    words = pickle.load(open(f'glove/6B.50_words.pkl', 'rb'))
    word2idx = pickle.load(open(f'glove/6B.50_idx.pkl', 'rb'))
    glove = {w: vectors[word2idx[w]] for w in words}
    max_sent = 0
    for sent in x_train:
        words = word_tokenize(sent)
        if len(words) > max_sent:
            max_sent = len(words)
    max_sent += 5
    print("Maximum Length of Sentence: ", max_sent)
    random_mat = np.random.normal(scale=0.6, size=(50))
    for i in range(len(x_train)):
        words = word_tokenize(x_train[i])
        emb_sent = np.zeros((max_sent, 50))
        for j in range(min(max_sent,len(words))):
            try:
                emb_sent[j] = glove[words[j]]
            except KeyError:
                emb_sent[j] = random_mat

        x_train[i] = emb_sent
    
    for i in range(len(x_dev)):
        words = word_tokenize(x_dev[i])
        emb_sent = np.zeros((max_sent, 50))
        for j in range(min(max_sent,len(words))):
            try:
                emb_sent[j] = glove[words[j]]
            except KeyError:
                emb_sent[j] = random_mat
        x_dev[i] = emb_sent
    del glove
    gc.collect()
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    return x_train, y_train, x_dev, y_dev, max_sent

def main():
    # Data loaders
    dataset_path = "../dataset/training.1600000.processed.noemoticon.csv"
    print("Open file:", dataset_path)
    DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "text"]
    df = pd.read_csv(dataset_path, encoding="ISO-8859-1" , names=DATASET_COLUMNS, nrows=800000)
    df2 = pd.read_csv(dataset_path, encoding="ISO-8859-1" , names=DATASET_COLUMNS, skiprows = 800000, nrows=800000)
    print("Dataset size:", 2*len(df))

    df.text = df.text.apply(lambda x: preprocess(x))
    df2.text = df2.text.apply(lambda x: preprocess(x))

    X = df["text"]
    y = df["target"]
    X, y = shuffle(X, y)
    X = X.tolist()
    y = y.tolist()

    X2 = df2["text"]
    y2 = df2["target"]
    X2, y2 = shuffle(X2, y2)
    X2 = X2.tolist()
    y2 = y2.tolist()
    for i in range(len(y2)):
        if y2[i] == 4:
            y2[i] = 1
        else:
            y2[i] = 0
    X = X + X2
    y = y + y2
    del df, X2, y2
    gc.collect()
    x_train, y_train, x_dev, y_dev, max_sent = preprocess_data(X,y)

    train_data = []
    for i in range(len(x_train)):
        train_data.append([x_train[i], y_train[i]])
    dev_data = []
    for i in range(len(x_dev)):
        dev_data.append([x_dev[i], y_dev[i]])
    batch_size = 128
    tr_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(dev_data, batch_size=batch_size, shuffle=False)
    #te_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Model
    model = CNN()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    #
    
    print('Number of float-valued parameters:', count_parameters(model))

    # Attempts to restore the latest checkpoint if exists
    print('Loading cnn...')
    model, start_epoch, stats = restore_checkpoint(model,
        config('cnn.checkpoint'))

    fig, axes = utils.make_cnn_training_plot()

    # Evaluate the randomly initialized model
    _evaluate_epoch(axes, tr_loader, va_loader, model, criterion, start_epoch,
        stats)

    # Loop over the entire dataset multiple times
    for epoch in range(start_epoch, config('cnn.num_epochs')):
        # Train model
        _train_epoch(tr_loader, model, criterion, optimizer)

        # Evaluate model
        _evaluate_epoch(axes, tr_loader, va_loader, model, criterion, epoch+1,
            stats)
        # _evaluate_clf(axes, va_loader, model, criterion, start_epoch,
        # stats)
        # Save model parameters
        save_checkpoint(model, epoch+1, config('cnn.checkpoint'), stats)
    # _evaluate_clf(axes, va_loader, model, criterion, start_epoch, stats)
    
    print('Finished Training')

    # Save figure and keep plot open
    utils.save_cnn_training_plot(fig)
    utils.hold_training_plot()

if __name__ == '__main__':
    main()
