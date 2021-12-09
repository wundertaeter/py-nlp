from .preprocessing import preprocesser
import numpy as np


#stopwords = set(w.rstrip() for w in open('/home/mori/project/data/stopwords.txt'))
preprocesser = preprocesser()


def clearLinks(list):
    for i in range(len(list)):
        string = list[i]
        if type(string) == tuple:
            string = string[0]
        list[i] = ''
        for line in string.split('\n'):
            if not 'www' in line and not 'http' in line:
                list[i] += line
    return list


class NaturalLanguageProcesser(object):
    def __init__(self, model, positive_news, negative_news, test_ratio=0.2, clear_links=False):
        self.model = model
        self.data = None
        self.train_data = None
        self.test_data = None
        self.positive_news = positive_news
        self.negative_news = negative_news
        self.clear_links = clear_links
        self.test_ratio = test_ratio

        if clear_links:
            positive_news = clearLinks(positive_news)
            negative_news = clearLinks(negative_news)

        self.get_data()

    def get_data(self):
        preprocesser.reset()
        positive_news = self.positive_news
        negative_news = self.negative_news

        print('positive len: ', len(positive_news))
        print('negative len: ', len(negative_news))

        if len(positive_news) > len(negative_news):
            positive_news = positive_news[:len(negative_news)]
        elif len(positive_news) < len(negative_news):
            negative_news = negative_news[:len(positive_news)]

        size = 2000
        if len(positive_news) > size:
            positive_news = positive_news[:size]
            negative_news = negative_news[:size]

        np.random.shuffle(positive_news)
        np.random.shuffle(negative_news)

        datas = (negative_news, positive_news)

        Xdata, Ydata = preprocesser.fit_transform(datas)
        self.data = [Xdata, Ydata]
        test_size = int(len(self.data[0]) * self.test_ratio)

        train_X = Xdata[:-test_size]
        train_Y = Ydata[:-test_size]
        self.train_data = [train_X, train_Y]

        test_X = Xdata[-test_size:]
        test_Y = Ydata[-test_size:]
        self.test_data = [test_X, test_Y]

    def set_model(self, model):
        self.model = model

    def fit(self):
        X, Y = self.train_data
        self.model.fit(X, Y)
        accuracy = self.model.score(X, Y)
        print("Train accuracy: {}".format(accuracy))
        return accuracy

    def score(self):
        assert self.test_data is not None, 'please fit first'
        X, Y = self.test_data
        accuracy = self.model.score(X, Y)
        print("Test accuracy : {}".format(accuracy))
        return accuracy

    def predict(self, data):
        if type(data) != list:
            data = [data]

        if self.clear_links:
            data = clearLinks(data)

        data = preprocesser.prepare(data)
        prediction = self.model.predict(data)
        return int(prediction)
