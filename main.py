#from argparse import ArgumentParser
import logging
import math
import random
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import import_data


def main():
    #device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")

    # Load the data
    train = import_data.load_file('wiki.train.txt')
    valid = import_data.load_file('wiki.valid.txt')
    test = import_data.load_file('wiki.test.txt')

    # Create the vocabulary

    # Define parameters

    # Execute the RNN model

if __name__ == '__main__':
    main()