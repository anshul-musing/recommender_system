import pandas as pd
import numpy as np
import pyomo.environ as aml


class MF(object):

    def __init__(self, dim):
        self.mapping = None
        self.dim = dim
        self.rhat = None


    def fit(self, interactions, ints_clean, ints_pos, ints_neg, method):
        self._build_map(interactions)
        try:
            exec('self._fit_{m}(ints_clean, ints_pos, ints_neg)'.format(m=method))
        except ValueError:
            print('Invalid mf algorithm')


    def predict(self, payload):
        cid = self._get_cust_id(payload)
        scores = self.rhat[cid,:]
        item_scores = {}
        for j in range(len(scores)):
            item_scores[self.mapping['id_to_item'][j]] = scores[j]
        
        sort_id = np.argsort(-scores)
        items = [self.mapping['id_to_item'][j] for j in sort_id]
        return item_scores, items


    def _build_map(self, interactions):
        custs = interactions['user'].unique()
        cid = np.arange(len(custs))

        items = interactions['product'].unique()
        iid = np.arange(len(items))

        mappings = {}
        mappings['cust_to_id'] = dict(zip(custs, cid))
        mappings['id_to_item'] = dict(zip(iid, items))

        self.mapping = mappings


    def _fit_svd(self, a, apos, aneg):
        k = self.dim
        u, s, v = np.linalg.svd(a, full_matrices=False)
        a_approx_svd = np.dot(u[:,0:k], np.dot(np.diag(s[0:k]), v[:k,:]))
        self.rhat = a_approx_svd


    def _fit_nmf(self, a, apos, aneg):
        from sklearn.decomposition import NMF

        nmf = NMF(n_components=self.dim, init='nndsvd', verbose=0)

        Wpos = nmf.fit_transform(apos)
        Hpos = nmf.components_
        Wneg = nmf.fit_transform(aneg)
        Hneg = nmf.components_

        a_approx_nmf = np.dot(Wpos,Hpos) - np.dot(Wneg,Hneg)
        self.rhat = a_approx_nmf


    def _fit_lda(self, a, apos, aneg):
        from sklearn.decomposition import LatentDirichletAllocation

        lda = LatentDirichletAllocation(n_components=self.dim, max_iter=1000, random_state=618, verbose=0)

        ldapos = lda.fit(apos)
        Hpos = ldapos.components_
        Wpos = ldapos.transform(apos)

        ldaneg = lda.fit(aneg)
        Hneg = ldaneg.components_
        Wneg = ldaneg.transform(aneg)

        a_approx_lda = np.dot(Wpos,Hpos) - np.dot(Wneg,Hneg)
        self.rhat = a_approx_lda


    def _fit_als(self, a, apos, aneg):
        k = self.dim
        num_custs, num_items = a.shape
        np.random.seed(618)
        P = np.random.rand(num_custs, k)
        Q = np.random.rand(num_items, k)

        Popt, Qopt = self._als_loop(num_custs, num_items, a, P, Q, k)
        a_approx_als = np.dot(Popt, Qopt.T)
        self.rhat = a_approx_als


    def _als_loop(self, nc, ni, a, P, Q, dim, num_iter=5000, alpha=0.002, beta=0.002):
        Q = Q.T
        for iter in range(num_iter):
            
            for c in range(nc):
                for i in range(ni):
                    if a[c,i] > 0 or a[c,i] < 0:
                        eci = a[c,i] - np.dot(P[c,:],Q[:,i])
                        for k in range(dim):
                            P[c,k] = P[c,k] + alpha * (2 * eci * Q[k,i] - beta * P[c,k])
                            Q[k,i] = Q[k,i] + alpha * (2 * eci * P[c,k] - beta * Q[k,i])

            e = 0
            for c in range(nc):
                for i in range(ni):
                    if a[c,i] > 0 or a[c,i] < 0:
                        e = e + pow(a[c,i] - np.dot(P[c,:],Q[:,i]), 2)
                        for k in range(dim):
                            e = e + (beta/2) * (pow(P[c,k],2) + pow(Q[k,i],2))

            if iter%100 == 0:
                print('Iteration ' + str(iter) + ' error: ' + str(e))

            if e < 0.001:
                break

        return P, Q.T


    def _fit_xmf(self, a, apos, aneg):
        k = self.dim
        num_custs, num_items = a.shape
        np.random.seed(618)
        pinit = np.random.rand(num_custs, k)
        qinit = np.random.rand(k, num_items)

        ins = self._xmf_model(a, pinit, qinit, k)
        solver = aml.SolverFactory('ipopt', keepfiles=False)
        solver.options['max_cpu_time'] = 180
        solver.solve(ins, tee=False)

        a_approx_xmf = np.zeros((num_custs, num_items))
        for c in ins.custs:
            for i in ins.items:
                a_approx_xmf[c,i] = ins.bc[c].value + ins.bi[i].value \
                                    + sum(ins.p[c,k].value * ins.q[k,i].value for k in ins.emb)

        self.rhat = a_approx_xmf


    def _xmf_model(self, a, pinit, qinit, dim, beta=0.002):
        # model instance
        m = aml.ConcreteModel()

        # sets
        m.custs = range(a.shape[0])
        m.items = range(a.shape[1])
        m.emb = range(dim)

        # variables
        m.bc = aml.Var(m.custs, domain=aml.Reals)
        m.bi = aml.Var(m.items, domain=aml.Reals)

        def initialize_p(m, c, k):
            return pinit[c,k]
        m.p = aml.Var(m.custs, m.emb, domain=aml.Reals, initialize=initialize_p)

        def initialize_q(m, k, i):
            return qinit[k,i]
        m.q = aml.Var(m.emb, m.items, domain=aml.Reals, initialize=initialize_q)

        # objective
        def obj_rule(m):
            e = 0.0
            for c in m.custs:
                for i in m.items:
                    if a[c,i] > 0 or a[c,i] < 0:
                        r = m.bc[c] + m.bi[i] + sum(m.p[c,k] * m.q[k,i] for k in m.emb)
                        e = e + pow(a[c,i] - r, 2)
                        e = e + (beta/2) * (pow(m.bc[c],2) + pow(m.bi[i],2) + sum(pow(m.p[c,k],2) + pow(m.q[k,i],2) for k in m.emb))

            return e
        m.obj = aml.Objective(rule=obj_rule, sense=aml.minimize)

        return m


    def _get_cust_id(self, payload):
        return self.mapping['cust_to_id'][payload['user']]

