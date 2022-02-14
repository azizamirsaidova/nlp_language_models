import import_data_RNN

# Create the vocabulary dictionary and set indexes, where Key: word, value: index
def create_vocabulary(file):
    f = open(file, 'r')
    file_1 = f.read()
    file_1 = file_1.lower()
    corpus = file_1.split(" ")
    vocab = {}
    count = 0
    # Review all the corpus to get unique words and give an index
    for word in corpus:
        if word not in vocab.keys():
            vocab[word] = count
            count += 1
    return vocab

# Transform the corpus from words to indexes tokens
def corpus_to_index(corpus, vocab):
    new_corpus = []
    for sentence in corpus:
        aux_sentence = []
        new_corpus.append(aux_sentence)
        for word in sentence:
            if word in vocab:
                index = vocab[word]
                aux_sentence.append(index)
            else:
                index = "<unk>"
                aux_sentence.append(index)
    return new_corpus

def main():
    vocab = create_vocabulary('wiki.train.txt')
    print(vocab)
    corpus = import_data.load_file('wiki.train.txt')
    corpus_index = corpus_to_index(corpus, vocab)
    print(corpus_index)

if __name__ == '__main__':
    main()