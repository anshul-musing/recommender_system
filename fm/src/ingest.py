import pandas as pd
import os


class Ingestor(object):
    def __init__(self, path):
        self.path = path


    def perform(self):
        fname = os.path.join(self.path, 'historical_swipes.csv')
        swipes = pd.read_csv(fname)

        fname = os.path.join(self.path, 'styles_tags.csv')
        styles = pd.read_csv(fname)

        return swipes, styles
