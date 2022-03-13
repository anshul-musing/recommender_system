import pandas as pd
import numpy as np
from mf.src.ingest import Ingestor
from mf.src.preprocess import Preprocessor
from mf.src.mf import MF


class Factorizer(object):

    def __init__(self):
        self.mf = None


    def fit(self, method='xmf'):
        print('ingesting ... ')
        ing = Ingestor(path='data/')
        ratings = ing.perform()

        print('preprocessing ... ')
        pp = Preprocessor()
        ints_clean, ints_pos, ints_neg = pp.fit_transform(ratings)

        print("fitting matrix factorization ... ")
        self.mf = MF(dim=3)
        self.mf.fit(ratings, ints_clean, ints_pos, ints_neg, method=method)


    def predict(self, payload):
        return self.mf.predict(payload)
