import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler
from gensim.models import Word2Vec
from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline

import nltk
from nltk import word_tokenize
nltk.download('stopwords')
from nltk.tokenize import RegexpTokenizer
from nltk.util import ngrams
from sklearn.pipeline import Pipeline
import joblib



class Model():
    def __init__(self, model, cat_df):
        self.model = model
        self.cat = cat_df
        
    def fit():
        pass
    
    def predict(self, X):
        category_id = self.model.predict(X)
        category_title = self.cat[self.cat['category_2'] == category_id[0]]['category_title']
        category_path = self.cat[self.cat['category_2'] == category_id[0]]['category_path']
        return {'category_id' : category_id[0], 'category_title': category_title, 'category_path': category_path}


class NormolizeText():
    def __init__(self, w2v):
        self.w2v = w2v
        
    def preprocess_text(self, text):
        stop_word_list = nltk.corpus.stopwords.words('russian')
        tokeniser = RegexpTokenizer("[A-Za-zА-Яа-я]+")
        tokens = tokeniser.tokenize(text)
        
        tokens_lower = [t.lower() for t in tokens]
        tokens_clean = [t for t in tokens_lower if t not in stop_word_list]
        return ' '.join(tokens_clean)

    def vectorize(self, text):
        words = [self.w2v.wv[w] for w in text.split() if w in self.w2v.wv]
        if len(words) > 0:
            return np.mean(words, axis=0)
        return np.zeros(50)

    def fit():
        pass
        
    def transform(self, x):
        preprocessed = self.preprocess_text(x)
        features_x = self.vectorize(preprocessed)
        return features_x


pipeline = joblib.load('model.joblib')

print(pipeline.predict('Массажер для ног'))

print(pipeline.predict('Контейнер с дозатором'))

print(pipeline.predict('Bluetooth'))
