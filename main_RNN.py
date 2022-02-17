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

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, tied, dropout):
        super(Rnn, self).__init__()
        self.tied = tied
        if not tied:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers, dropout=dropout)
        self.fc1 = nn.Linear(hidden_dim, vocab_size)

    def get_embedded(self, word_indexes):
        if self.tied:
            return self.fc1.weight.index_select(0, word_indexes)
        else:
            return self.embedding(word_indexes)

    def init_weights(self):
        init = 0.1
        self.encoder.weight.data.uniform_(-init, init)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init, init)

    def forward(self, packed_sents):
        embedded_sents = nn.utils.rnn.PackedSequence(self.get_embedded(packed_sents.data), packed_sents.batch_sizes)
        out_packed_sequence, input_sizes = self.rnn(embedded_sents)
        out = self.fc1(out_packed_sequence.data)
        last_seq = out[-61:]
        return F.log_softmax(last_seq, dim=1)

def batches(data, batch_size):
    batches_list = []
    for i in range(0, len(data)):
        batch = data[i:i + batch_size]
        batches_list.append(batch)
    random.shuffle(data)
    return batches_list


def time_step(model, data, device):
    '''  We feed 'x' with 30 words, to predict word 'y' number 31 '''
    data_tensor = torch.tensor(data)
    x = nn.utils.rnn.pack_sequence([data_tensor[i : i+29] for i in range(0, len(data)-30)])
    y = nn.utils.rnn.pack_sequence([data_tensor[i+29 : i+30] for i in range(0, len(data)-30)])

    if device.type == 'cuda':
        x, y = x.cuda(), y.cuda()
    out = model(x)
    loss = F.nll_loss(out, y.data)
    return out, loss, y

def train_epoch(data, model, optimizer, args, device):
    model.train()
    batch_count = 0
    perplexity_overall = 0
    for batch in batches(data, args.batch_size):
        if batch_count > args.batch_count:
            break
        model.zero_grad()
        out, loss, y = time_step(model, batch, device)
        loss.backward()
        optimizer.step()
        if batch_count <= args.batch_count:
            # Calculate perplexity.
            prob = out.exp()[torch.arange(0, y.data.shape[0], dtype=torch.int64), y.data]
            perplexity = 2 ** prob.log2().neg().mean().item()
            perplexity_overall = perplexity_overall + perplexity
            logging.info("\tBatch %d, loss %.3f, perplexity %.2f", batch_count, loss.item(), perplexity)
            batch_count += 1

    perplexity_epoch = perplexity_overall / batch_count

    return perplexity_epoch


def perplexity_eval(data, model, batch_size, device):
    model.eval()
    with torch.no_grad():
        entropy_sum = 0
        word_count = 0
        batch_count = 0
        for batch in batches(data, batch_size):
            if batch_count > 10:
                break
            out, _, y = time_step(model, batch, device)
            prob = out.exp()[torch.arange(0, y.data.shape[0], dtype=torch.int64), y.data]
            entropy_sum += prob.log2().neg().sum().item()
            word_count += y.data.shape[0]
            batch_count += 1
    return 2 ** (entropy_sum / word_count)

def parse_args(args):
    argp = ArgumentParser(description=__doc__)
    argp.add_argument("--logging", choices=["INFO", "DEBUG"],default="INFO")
    argp.add_argument("--embedding-dim", type=int, default=100)
    argp.add_argument("--untied", action="store_true")
    argp.add_argument("--num-hidden", type=int, default=100)
    argp.add_argument("--num-layers", type=int, default=2)
    argp.add_argument("--num-dropout", type=float, default=0.0)
    argp.add_argument("--epochs", type=int, default=20)
    argp.add_argument("--batch-size", type=int, default=91)
    argp.add_argument("--lr", type=float, default=0.001)
    argp.add_argument("--no-cuda", action="store_true")
    argp.add_argument("--batch-count", type=int, default=20)
    return argp.parse_args(args)

def plot_perplexity(perplexity_train, perplexity_valid, perplexity_test):
    plt.plot(perplexity_train)
    plt.plot(perplexity_valid)
    plt.plot(perplexity_test)
    plt.title('RNN Model perplexity')
    plt.ylabel('perplexity')
    plt.xlabel('epoch')
    plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
    plt.legend(['train', 'validation', 'test'], loc='upper left')
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

    # Transform WORD tokens from words to indexes
    train_indexes_word = vocabulary_RNN.word_to_index(train, vocab)
    valid_indexes_word = vocabulary_RNN.word_to_index(valid, vocab)
    test_indexes_word = vocabulary_RNN.word_to_index(test, vocab)

    model = Rnn(len(vocab), args.embedding_dim, args.num_hidden, args.num_layers, not args.untied, args.num_dropout).to(device)
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr)

    perplexity_train = []
    perplexity_valid = []
    perplexity_test = []

    for epoch_ind in range(args.epochs):
        logging.info("Training epoch %d", epoch_ind)

        perp_train = train_epoch(train_indexes_word, model, optimizer, args, device)
        perp_valid = perplexity_eval(valid_indexes_word, model, args.batch_size, device)
        perp_test = perplexity_eval(test_indexes_word, model, args.batch_size, device)

        perplexity_train.append(perp_train)
        perplexity_valid.append(perp_valid)
        perplexity_test.append(perp_test)

        logging.info("Train perplexity: %.1f", perp_train)
        logging.info("Validation perplexity: %.1f", perp_valid)
        logging.info("Test perplexity: %.1f", perp_test)

    plot_perplexity(perplexity_train, perplexity_valid, perplexity_test)

if __name__ == '__main__':
    start_time = datetime.now()
    main()
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))
