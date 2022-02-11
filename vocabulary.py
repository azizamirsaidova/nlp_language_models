import import_data

# Create the vocabulary dictionary and set indexes, where Key: word, value: index
def create_vocabulary(corpus):
    vocab = {}
    count = 0
    # Review all the corpus to get unique words and give an index
    for sentence in corpus:
        for word in sentence:
            if word not in vocab.keys():
                vocab[word] = count
                count += 1
    return vocab

# Transform the corpus from words to indexes tokens
def corpus_to_index(corpus, vocab):
    new_corpus = []
    for sentence in corpus:
        for word in sentence:
            index = vocab[word]
            new_corpus.append(index)
    return new_corpus

def main():
    train = import_data.load_file('wiki.train.txt')
    vocab = create_vocabulary(train)
    print(vocab)

if __name__ == '__main__':
    main()