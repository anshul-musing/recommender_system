import pandas as pd
import os


class Ingestor(object):
    def __init__(self, path):
        self.path = path


    def perform(self):
        fname = os.path.join(self.path, 'ratings.csv')
        ratings = pd.read_csv(fname)

        return ratings
