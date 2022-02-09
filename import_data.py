import os
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize

# Load the file and tokenize in sentences, and also each sentence tokenized in words
def load_file(file):
    f = open(file, 'r')
    file_1 = f.read()
    sentences = nltk.tokenize.sent_tokenize(file_1)
    for sentence in sentences:
        aux = sentence.split(" ")
        sentences[sentences.index(sentence)] = aux
    return sentences

'''
def load(path, index):
    """ Loads the wikitext2 data at the given path using
    the given index (maps tokens to indices). Returns
    a list of sentences where each is a list of token
    indices.
    """
    start = index.add(SENT_START)
    sentences = []
    with open(path, "r") as f:
        for paragraph in f:
            for sentence in paragraph.split(" . "):
                tokens = sentence.split()
                if not tokens:
                    continue
                sentence = [index.add(SENT_START)]
                sentence.extend(index.add(t.lower()) for t in tokens)
                sentence.append(index.add(SENT_END))
                sentences.append(sentence)

    return sentences

def main():

    index = Vocab()
    for part in ("train", "valid", "test"):
        print("Processing", part)
        sentences = load(path(part), index)
        print("Found", sum(len(s) for s in sentences),
              "tokens in", len(sentences), "sentences")
    print("Found in total", len(index), "tokens")
'''