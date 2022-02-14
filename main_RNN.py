#from argparse import ArgumentParser
import logging
import math
import random
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import import_data_RNN
import vocabulary_RNN

class Rnn(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, layers, tied, dropout):
        super(Rnn, self).__init__()
        self.tied = tied
        if not tied:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(hidden_dim, vocab_size)

    def embeddings(self, word_indexes):
        if self.tied:
            return self.fc1.weight.index_select(0, word_indexes)
        else:
            return self.embedding(word_indexes)

    def forward(self, packed_sents):
            """ Takes a PackedSequence of sentences tokens that has T tokens
            belonging to vocabulary V. Outputs predicted log-probabilities
            for the token following the one that's input in a tensor shaped
            (T, |V|).
            """
            embedded_sents = nn.utils.rnn.PackedSequence(
                self.embeddings(packed_sents.data), packed_sents.batch_sizes)
            out_packed_sequence, _ = self.gru(embedded_sents)
            out = self.fc1(out_packed_sequence.data)
            return F.log_softmax(out, dim=1)

def batches(data, batch_size):
    """ Yields batches of sentences from 'data', ordered on length. """
    random.shuffle(data)
    for i in range(0, len(data), batch_size):
        sentences = data[i:i + batch_size]
        sentences.sort(key=lambda l: len(l), reverse=True)
        yield [torch.LongTensor(s) for s in sentences]

def step(model, sents, device):
    """ Performs a model inference for the given model and sentence batch.
    Returns the model otput, total loss and target outputs. """
    x = nn.utils.rnn.pack_sequence([s[:-1] for s in sents])
    y = nn.utils.rnn.pack_sequence([s[1:] for s in sents])
    if device.type == 'cuda':
        x, y = x.cuda(), y.cuda()
    out = model(x)
    loss = F.nll_loss(out, y.data)
    return out, loss, y

def train_epoch(data, model, optimizer, batch_size, device):
    """ Trains a single epoch of the given model. """
    model.train()
    #log_timer = LogTimer(5)
    for batch_ind, sents in enumerate(batches(data, batch_size)):
        model.zero_grad()
        out, loss, y = step(model, sents, device)
        loss.backward()
        optimizer.step()
        if batch_ind == 0:
            # Calculate perplexity.
            prob = out.exp()[
                torch.arange(0, y.data.shape[0], dtype=torch.int64), y.data]
            perplexity = 2 ** prob.log2().neg().mean().item()
            logging.info("\tBatch %d, loss %.3f, perplexity %.2f",
                         batch_ind, loss.item(), perplexity)

def evaluate(data, model, batch_size, device):
    """ Perplexity of the given data with the given model. """
    model.eval()
    with torch.no_grad():
        entropy_sum = 0
        word_count = 0
        for sents in batches(data, batch_size):
            out, _, y = step(model, sents, device)
            prob = out.exp()[
                torch.arange(0, y.data.shape[0], dtype=torch.int64), y.data]
            entropy_sum += prob.log2().neg().sum().item()
            word_count += y.data.shape[0]
    return 2 ** (entropy_sum / word_count)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the data
    train = import_data_RNN.load_file('wiki.train.txt')
    valid = import_data_RNN.load_file('wiki.valid.txt')
    test = import_data_RNN.load_file('wiki.test.txt')

    # Create the vocabulary
    vocab = vocabulary_RNN.create_vocabulary('wiki.train.txt')

    # Transform tokens from words to indexes
    train_indexes = vocabulary_RNN.corpus_to_index(train, vocab)
    valid_indexes = vocabulary_RNN.corpus_to_index(valid, vocab)
    test_indexes = vocabulary_RNN.corpus_to_index(test, vocab)

    # Define parameters
    embedding_dim = 100
    hidden_dim = 1
    layers = 2
    untied = True
    gru_dropout = 0
    learning_rate = 0.001
    epochs = 4
    batches = 20
    batch_size = 200

    # Execute the RNN model
    model = Rnn(len(vocab), embedding_dim, hidden_dim, layers, untied, gru_dropout)
    optimizer = optim.RMSprop(model.parameters(), lr = learning_rate)

    for epoch_ind in range(epochs):
        logging.info("Training epoch %d", epoch_ind)
        train_epoch(train_indexes, model, optimizer, batch_size, device)
        logging.info("Validation perplexity: %.1f", evaluate(valid_indexes, model, batch_size, device))
    logging.info("Test perplexity: %.1f", evaluate(test_indexes, model, batch_size, device))

if __name__ == '__main__':
    main()