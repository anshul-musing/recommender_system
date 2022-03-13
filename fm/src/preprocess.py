import numpy as np
import pandas as pd
import re


class Preprocessor(object):

    def __init__(self):
        pass


    def fit_transform(self, interactions, item_side_data):
        ints_clean = self.perform_ints(interactions)
        usd_clean = self.perform_usd(interactions)
        isd_clean = self.perform_isd(interactions, item_side_data)
        return ints_clean, usd_clean, isd_clean


    def perform_ints(self, interactions):
        '''
        Reshapes the interaction dataframe to a numpy array
        '''
        custs = interactions['User_ID'].unique()
        items = interactions['Style_ID'].unique()
        data = interactions['Interaction'].values
        ints = np.array(data).reshape(len(custs), len(items))

        return ints


    def perform_usd(self, interactions):
        '''
        Performs target encoding for different user data features
        Assign an identity columns to each user
        Return as an array
        '''
        usd = pd.Series(interactions['User_ID'].unique()).reset_index()
        usd = usd.assign(id=1)
        usd = usd.assign(dummy_feat=0)
        return usd[['id','dummy_feat']].values


    def perform_isd(self, ints, item_side_data):
        '''
        Performs target encoding for different side data features
        Assign an identity columns to each item
        Return as an array
        '''
        isd = item_side_data.copy()

        # mean interaction score for each style
        isd_ints = ints.groupby('Style_ID')['Interaction'].apply(lambda x: abs(x).mean()).reset_index()
        isd = isd.drop('Tags', axis=1)
        isd = isd.merge(isd_ints, on='Style_ID', how='left')

        # mean target encoding for each feature
        cols = isd.columns.difference(['Style_ID', 'Interaction']).tolist()
        for c in cols:
            cmean = isd.groupby(c)['Interaction'].mean().reset_index().rename(columns={'Interaction':c+'_te'})
            isd = isd.merge(cmean, on=c, how='left')
            isd = isd.drop(c, axis=1)

        isd = isd.drop('Interaction', axis=1)

        # consolidate target encoding scores for each style
        isd_cons = isd.groupby('Style_ID').mean().reset_index()

        # assign the identity column
        isd_cons.insert(loc=1, column='id', value=1)

        # return the array
        col_list = ['id'] + [a+'_te' for a in cols]

        return isd_cons[col_list].values

