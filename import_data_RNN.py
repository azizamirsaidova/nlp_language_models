import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize

# Load the file and tokenize in sentences, and also each sentence tokenized in words
def load_file(file):
    f = open(file, 'r')
    file_1 = f.read()
    file_1 = file_1.lower()
    sentences = nltk.tokenize.sent_tokenize(file_1)
    for sentence in sentences:
        aux = sentence.split(" ")
        sentences[sentences.index(sentence)] = aux
    return sentences


