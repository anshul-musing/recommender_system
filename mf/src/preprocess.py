import numpy as np
import pandas as pd
import re


class Preprocessor(object):

    def __init__(self):
        pass


    def fit_transform(self, interactions):
        ints_clean, ints_pos, ints_neg = self.perform_ints(interactions)
        return ints_clean, ints_pos, ints_neg


    def perform_ints(self, interactions):
        custs = interactions['user'].unique()
        items = interactions['product'].unique()
        data = interactions['interaction'].values
        ints = np.array(data).reshape(len(custs), len(items))

        ints_pos = np.copy(ints)
        ints_neg = np.copy(ints)

        ints_pos[ints_pos<0] = 0
        ints_neg[ints_neg>0] = 0
        ints_neg = -ints_neg

        return ints, ints_pos, ints_neg
