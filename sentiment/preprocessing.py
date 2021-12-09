import nltk
import numpy as np
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords as stopWords
from sklearn.utils import shuffle


wordnet_lemmatizer = WordNetLemmatizer()

def get_names(s):
    chunked = ne_chunk(pos_tag(word_tokenize(s)))
    continuous_chunk = []
    current_chunk = []
    for i in chunked:
            if type(i) == Tree:
                    current_chunk.append(" ".join([token for token, pos in i.leaves()]))
            elif current_chunk:
                    named_entity = " ".join(current_chunk)
                    if named_entity not in continuous_chunk:
                            continuous_chunk.append(named_entity.lower())
                            current_chunk = []
            else:
                    continue
    return continuous_chunk

def remove_names(s, names):
    for name in names:
        if s.find(name) > -1:
            s = s[:s.find(name)] + s[s.find(name)+len(name):] # remove Persons and Company Names
    return s

def tokenize(s, stopwords):
    if type(s) == tuple:
        s = s[0]
    s = str(s)
    tokens = []
    #names = tools.get_names(s)
    #tokens.extend(names)
    s = s.lower()  # downcase
    #s = tools.remove_names(s, names) 
    for token in nltk.tokenize.word_tokenize(s): # split string into words (tokens)
        if len(token) < 2:  # remove short words, they're probably not useful
            continue
        token = wordnet_lemmatizer.lemmatize(token) # put words into base form     
        if token in stopwords: # remove stopwords
            continue
        append = True
        for char in token: 
            if not char.isalpha():
                append = False
                break
        if append: # remove words that doesn't make sense
            tokens.append(token)
    return tokens

def vectorize(tokens, word_index_map):
    x = []
    for token in tokens:
        if token in word_index_map:
            i = word_index_map[token]
            x.append(i)
    return x

def oneHotEncoder(X, lenth):
    data = np.zeros((len(X), lenth))
    i = 0
    for x in X:
        for index in x:
            data[i][index] += 1
        i += 1
    return data

def oneHotEncoder_y(Y, lenth):
    data = np.zeros((len(Y), lenth))
    i = 0
    for y in Y:
        data[i][y] = 1
        i += 1
    return data


class preprocesser(object):

    def __init__(self, stopwords=[]):
        if len(stopwords) == 0:
            stopwords = stopWords.words('english')
        self.stopwords = set(stopwords)
        self.current_index = 0
        self.word_index_map = {}
        self.index_word_map = {}
        self.datas_tokenized = []

    def fit_transform(self, datas):
        self.fit(datas)
        X, Y = self.transform()
        return oneHotEncoder(X, self.current_index), np.array(Y)

    def fit(self, datas):
        for data in datas:
            tokenized = []
            for example in data:
                tokens = tokenize(example, self.stopwords)
                if len(tokens) < 5:
                    continue
                tokenized.append(tokens)
                for token in tokens:
                    if token not in self.word_index_map:
                        self.word_index_map[token] = self.current_index
                        self.index_word_map[self.current_index] = token
                        self.current_index += 1
            self.datas_tokenized.append(tokenized)

    def transform(self):
        if len(self.word_index_map) == 0:
            print('NO DATA!\nMABY NOT FITTET JET?')
            return

        data = []
        i = 0
        for data_tokenized in self.datas_tokenized:
            for tokens in data_tokenized:
                data.append((vectorize(tokens, self.word_index_map), i))
            i += 1

        data = shuffle(data)

        X = [tuple[0] for tuple in data]
        Y = [tuple[1] for tuple in data]

        return X, Y
    
    def prepare(self, data):
        if len(self.word_index_map) == 0:
            print('NO DATA!\nMABY NOT FITTET JET?')
            return
        
        for i in range(len(data)):
            tokens = tokenize(data[i], self.stopwords)
            data[i] = vectorize(tokens, self.word_index_map)

        
        return oneHotEncoder(data, self.current_index) 
    
    def reset(self):
        self.current_index = 0
        self.word_index_map = {}
        self.index_word_map = {}
        self.datas_tokenized = []
