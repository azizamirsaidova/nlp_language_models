from argparse import ArgumentParser
import logging
import math
import random
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import vocabulary_RNN
import import_data_RNN

from datetime import datetime
import matplotlib.pyplot as plt


class Rnn(nn.Module):
    """ A language model RNN with GRU layer(s). """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, gru_layers, tied, dropout):
        super(Rnn, self).__init__()
        self.tied = tied
        if not tied:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, gru_layers,
                          dropout=dropout)
        self.fc1 = nn.Linear(hidden_dim, vocab_size)

    def get_embedded(self, word_indexes):
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
            self.get_embedded(packed_sents.data), packed_sents.batch_sizes)
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
    ''' We feed x with 30 words, to predict word y number 31 '''
    x_list = []
    y_list = []
    for i in range(sents):
        x_aux = sents[i : i+30]
        y_aux = sents[i+30 : i + 31]
        x_list.append(x_aux)
        y_list.append(y_aux)

    x = nn.utils.rnn.pack_sequence(x_list)
    y = nn.utils.rnn.pack_sequence(y_list)
    #x = nn.utils.rnn.pack_sequence([s[:-1] for s in sents])
    #y = nn.utils.rnn.pack_sequence([s[1:] for s in sents])
    if device.type == 'cuda':
        x, y = x.cuda(), y.cuda()
    out = model(x)
    loss = F.nll_loss(out, y.data)
    return out, loss, y


def train_epoch(data, model, optimizer, args, device):
    """ Trains a single epoch of the given model. """
    model.train()
    for batch_ind, sents in enumerate(batches(data, args.batch_size)):
        if batch_ind > 5:
            break
        model.zero_grad()
        out, loss, y = step(model, sents, device)
        loss.backward()
        optimizer.step()
        if batch_ind <= 5:
            # Calculate perplexity.
            prob = out.exp()[
                torch.arange(0, y.data.shape[0], dtype=torch.int64), y.data]
            perplexity = 2 ** prob.log2().neg().mean().item()
            logging.info("\tBatch %d, loss %.3f, perplexity %.2f",
                         batch_ind, loss.item(), perplexity)


def perplexity_eval(data, model, batch_size, device):
    model.eval()
    with torch.no_grad():
        entropy_sum = 0
        word_count = 0
        for sents in batches(data, batch_size):
            out, _, y = step(model, sents, device)
            prob = out.exp()[torch.arange(0, y.data.shape[0], dtype=torch.int64), y.data]
            entropy_sum += prob.log2().neg().sum().item()
            word_count += y.data.shape[0]
    return 2 ** (entropy_sum / word_count)


def parse_args(args):
    argp = ArgumentParser(description=__doc__)
    argp.add_argument("--logging", choices=["INFO", "DEBUG"],default="INFO")
    argp.add_argument("--embedding-dim", type=int, default=100)
    argp.add_argument("--untied", action="store_true")
    argp.add_argument("--gru-hidden", type=int, default=100)
    argp.add_argument("--gru-layers", type=int, default=2)
    argp.add_argument("--gru-dropout", type=float, default=0.0)
    argp.add_argument("--epochs", type=int, default=5)
    argp.add_argument("--batch-size", type=int, default=512)
    argp.add_argument("--lr", type=float, default=0.001)
    argp.add_argument("--no-cuda", action="store_true")
    return argp.parse_args(args)

def plot_perplexity(perplexity_valid, perplexity_test):
    #plt.plot(perplexity_train)
    plt.plot(perplexity_valid)
    plt.plot(perplexity_test)
    plt.title('RNN Model perplexity')
    plt.ylabel('perplexity')
    plt.xlabel('epoch')
    plt.xticks([0, 1, 2, 3, 4])
    plt.legend(['validation', 'test'], loc='upper left')
    plt.show()

def main(args=sys.argv[1:]):
    args = parse_args(args)
    logging.basicConfig(level=args.logging)

    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")

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

    model = Rnn(len(vocab), args.embedding_dim,
                  args.gru_hidden, args.gru_layers,
                  not args.untied, args.gru_dropout).to(device)
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr)

    #perplexity_train = []
    perplexity_valid = []
    perplexity_test = []

    for epoch_ind in range(args.epochs):
        logging.info("Training epoch %d", epoch_ind)
        train_epoch(train_indexes, model, optimizer, args, device)

        #perp_train = evaluate(train_indexes, model, args.batch_size, device)
        perp_valid = perplexity_eval(valid_indexes, model, args.batch_size, device)
        perp_test = perplexity_eval(test_indexes, model, args.batch_size, device)

        #perplexity_train.append(perp_train)
        perplexity_valid.append(perp_valid)
        perplexity_test.append(perp_test)

        #logging.info("Train perplexity: %.1f", perp_train)
        logging.info("Validation perplexity: %.1f", perp_valid)
        logging.info("Test perplexity: %.1f", perp_test)

    plot_perplexity(perplexity_valid, perplexity_test)



if __name__ == '__main__':
    start_time = datetime.now()
    main()
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))