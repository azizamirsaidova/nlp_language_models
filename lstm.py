# -*- coding: utf-8 -*-
"""lstm


"""

#LIBRARIES
import os
from io import open
import time
import math
import argparse
import torch.onnx

import torch
import torch.nn as nn
import torch.nn.functional as F

# 2. Load in the text data

class Dict_word_idx(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    # def __len__(self):
    #     return len(self.idx2word)

class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dict_word_idx()
        self.train = self.tokenize(path+ 'train.txt')
        self.valid = self.tokenize(path+ 'valid.txt')
        self.test = self.tokenize(path+ 'test.txt')

    def tokenize(self, path):
        """Tokenizes a text file."""
        # assert os.path.exists(path)
        # word to index mapping - replacting new lines with eos tokens
        # with open(path, 'r', encoding="utf8") as f:
        #     for line in f:
        #         words = line.split() + ['<eos>']
        #         for word in words:
        #             self.dictionary.add_word(word)

        # Tokenize file content
        #word to number mapping
        with open(path, 'r') as f:
            all_id = []
            for line in f:
                words = line.split() + ['<eos>']
                ids = []
                for t in words:
                    ids.append(self.dictionary.add_word(t))
                    #store in single pytorch tensor 
                all_id.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(all_id)

        return ids

# model_data_filepath = 'data/'

# corpus = Corpus(model_data_filepath + 'wikitext-2')

# 1. Define the model
# Here we define the LSTM model architecture, following the
# `model <https://github.com/pytorch/examples/blob/master/word_language_model/model.py>`_
# from the word language model example.
class LSTMModel(nn.Module):
    """LSTM model with encoder decoder"""

    def __init__(self, ntoken, ninp, num_hiddens, num_layers):
        super(LSTMModel, self).__init__()
        self.ntoken = ntoken
        self.encoder = nn.Embedding(ntoken, ninp)
        print("yes")
        self.drop = nn.Dropout(0.3)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.lstm = nn.LSTM(ninp, num_hiddens, num_layers, dropout=0.65)
        self.decoder = nn.Linear(num_hiddens, ntoken)

        self.init_weights()
        self.encoder.weight = self.decoder.weight
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers

    def init_weights(self):
        init = 0.1
        self.encoder.weight.data.uniform_(-init, init)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init, init)

    def forward(self, input, hidden):
        x = self.encoder(input)
        x = self.drop(x)
        x, hidden = self.lstm(x, hidden)
        x = self.drop(x)
        x = self.decoder(x)
        x = x.view(-1, self.ntoken)
        return F.log_softmax(x, dim=1), hidden


    def init_hidden(self, batch_sz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_layers, batch_sz, self.num_hiddens),
                weight.new_zeros(self.num_layers, batch_sz, self.num_hiddens))

    
# 3. Load the pre-trained model

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='/content/wiki.')
parser.add_argument('--emsize', type=int, default=100)
parser.add_argument('--num_hidden', type=int, default=100)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--nlayers', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--seed', type=int, default=222)
parser.add_argument('--log-interval', type=int, default=200)
parser.add_argument('--lr', type=float, default=10)
parser.add_argument('--bptt', type=int, default=30)
parser.add_argument('--dry-run', action='store_true')
parser.add_argument('--nhead', type=int, default=2)
parser.add_argument('--dropout', type=float, default=0.2)
args, unknown = parser.parse_known_args()
# torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#creating batches for the data 
def batch(data, batch_sz):
    no_batch = data.size(0) // batch_sz
    data = data.narrow(0, 0, no_batch * batch_sz)
    data = data.view(batch_sz, -1).t().contiguous()
    return data.to(device)

def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

def repackage_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

eval_batch_size = 10
corpus = Corpus(args.data)
train_data = batch(corpus.train, args.batch_size)
val_data = batch(corpus.valid, args.batch_size)
test_data = batch(corpus.test, args.batch_size)

# # Build the model
num_tokens = len(corpus.dictionary.idx2word)

model = LSTMModel(num_tokens, args.emsize, args.num_hidden, args.nlayers).to(device)
criterion = nn.NLLLoss()

# Training code
def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    print(total_loss)

    hidden = model.init_hidden(args.batch_size)
    
   
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
    # for batch, i in enumerate(range(0, train_data.size(0) - 1, 1)):    
        data, targets = get_batch(train_data, i)
        
        model.zero_grad()

        
        hidden = repackage_hidden(hidden)
        
        output, hidden = model(data, hidden)
        
        loss = criterion(output, targets)
        loss.backward()
        
        # gradient clipping 
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.20)
        for p in model.parameters():
            p.data.add_(p.grad, alpha=-args.lr)

        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            
            print('| epoch {} |  '
                    'loss {} | ppl {}'.format(
                epoch,  cur_loss, math.exp(cur_loss)))
            total_loss = 0
        if args.dry_run:
            break

def evaluate(data):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    loss = 0.
    # ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    
    with torch.no_grad():
        for i in range(0, data.size(0) - 1, args.bptt):
            
            x, targets = get_batch(data, i)
            
            output, hidden = model(x, hidden)
            
            hidden = repackage_hidden(hidden)
            actual_loss =criterion(output, targets).item()
            
            loss += actual_loss * len(x) 
            
            
    return loss / (len(data) - 1)

for epoch in range(1, args.epochs+1):
  train()
  val_loss = evaluate(val_data)
        
  print('| end of epoch {} | valid loss {} | '
                'valid ppl {}'.format(epoch,
                                           val_loss, math.exp(val_loss)))
  test_loss = evaluate(test_data)
        
  print('| end of epoch {} | test loss {} | '
                'test ppl {}'.format(epoch,
                                           test_loss, math.exp(val_loss)))

perplexity_train = [403,305,261,237,222,210,202,185,191,186,182,180,177,176,173,171,169,167,166,164]
perplexity_valid = [329,264,235,217,207,201,196,192,189,189,187,184,183,180,185,179,180,180,181,179]
perplexity_test = [329,264,235,217,207,201,196,192,189,189,187,184,183,180,185,179,180,180,181,179]
import matplotlib.pyplot as plt

def plot_perplexity(perplexity_train, perplexity_valid,perplexity_test):
    plt.plot(perplexity_train)
    plt.plot(perplexity_valid)
    plt.plot(perplexity_test)
    plt.title('RNN Model perplexity')
    plt.ylabel('perplexity')
    plt.xlabel('epoch')
    plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
    # plt.legend(['train', 'validation'], loc='upper left')
    plt.legend(['train', 'validation', 'test'], loc='upper left')
    plt.show()

# plot_perplexity(perplexity_train, perplexity_valid)

plot_perplexity(perplexity_train, perplexity_valid, perplexity_test)

