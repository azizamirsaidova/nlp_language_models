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

# Transform the corpus to indexes. List of sentences, with lists of words inside them.
def corpus_to_index(corpus, vocab):
    new_corpus = []
    for sentence in corpus:
        aux_sentence = []
        new_corpus.append(aux_sentence)
        for word in sentence:
            if word in vocab:
                index = vocab[word]
                aux_sentence.append(index)
    new_list = [x for x in new_corpus if len(x) >= 2]
    return new_list

# Transform the corpus to indexes. List of words.
def word_to_index(corpus, vocab):
    new_corpus = []
    for sentence in corpus:
        for word in sentence:
            if word in vocab:
                index = vocab[word]
                new_corpus.append(index)
    return new_corpus

def main():
    vocab = create_vocabulary('wiki.train.txt')
    corpus = import_data_RNN.load_file('wiki.train.txt')
    word_index = word_to_index(corpus, vocab)
    print(word_index)

if __name__ == '__main__':
    main()