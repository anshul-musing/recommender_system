import pandas as pd
import numpy as np
from swipe_recs_fm.src.ingest import Ingestor
from swipe_recs_fm.src.preprocess import Preprocessor
from swipe_recs_fm.src.fm import FM


class Factorizer(object):

    def __init__(self):
        self.fm = None


    def fit(self, verbose=0):
        print('ingesting ... ')
        ing = Ingestor(path='data/')
        swipes, styles = ing.perform()

        print('preprocessing ... ')
        pp = Preprocessor()
        ints_clean, usd_clean, isd_clean = pp.fit_transform(swipes, styles)

        print("fitting FM ... ")
        self.fm = FM(dim=3, verbose=verbose)
        self.fm.fit(swipes, ints_clean, usd_clean, isd_clean)


    def save(self):
        # save the predicted ratings
        fn = 'persisted/predicted_ratings.csv'
        np.savetxt(fn, self.fm.rhat, delimiter=",")

        # save the customer mapping
        df = pd.Series(self.fm.mapping['cust_to_id']).reset_index().rename(columns={'index':'cust',0:'id'})
        fn = 'persisted/customer_mapping.csv'
        df.to_csv(fn, index=False, encoding='utf-8')

        # save the item mapping
        df = pd.Series(self.fm.mapping['id_to_item']).reset_index().rename(columns={'index':'id',0:'item'})
        fn = 'persisted/item_mapping.csv'
        df.to_csv(fn, index=False, encoding='utf-8')

        # save the popularity scores
        scores = np.tanh([self.fm.model.b0.value + self.fm.model.bi[i].value for i in self.fm.model.items])
        pop_scores = {}
        for j in range(len(scores)):
            pop_scores[self.fm.mapping['id_to_item'][j]] = scores[j]
        df = pd.Series(pop_scores).reset_index().rename(columns={'index':'item',0:'score'})
        fn = 'persisted/popularity_score.csv'
        df.to_csv(fn, index=False, encoding='utf-8')

        print('\nsolutions saved as a csv')


    def predict(self, payload):

        # load the datasets
        rhat = np.genfromtxt('persisted/predicted_ratings.csv', delimiter=',')
        cust_df = pd.read_csv('persisted/customer_mapping.csv')
        item_df = pd.read_csv('persisted/item_mapping.csv')
        pop_df = pd.read_csv('persisted/popularity_score.csv')

        # convert to dictionaries
        cust_to_id = cust_df.set_index('cust').to_dict()['id']
        id_to_item = item_df.set_index('id').to_dict()['item']
        pop_scores = pop_df.set_index('item').to_dict()['score']

        # get customer id
        cid = self._get_cust_id(payload, cust_to_id)        
        if not cid is np.nan:
            scores = rhat[cid,:]
            item_scores = {}
            for j in range(len(scores)):
                item_scores[id_to_item[j]] = scores[j]        
        
        else: #popularity fallback
            scores = np.array(list(pop_scores.values()))
            item_scores = pop_scores

        sort_id = np.argsort(-scores)
        items = [id_to_item[j] for j in sort_id]

        return item_scores, items


    def _get_cust_id(self, payload, cust_to_id):
        if payload['user_id'] in cust_to_id:
            return cust_to_id[payload['user_id']]
        else:
            return np.nan
