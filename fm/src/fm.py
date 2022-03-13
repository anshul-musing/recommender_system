import pandas as pd
import numpy as np
import pyomo.environ as aml
from pyomo.environ import *


class FM(object):

    def __init__(self, dim, verbose=0):
        self.mapping = None
        self.dim = dim
        self.verbose = True if verbose else False
        self.model = None
        self.rhat = None


    def fit(self, interactions, ints_clean, usd_clean, isd_clean):
        self._build_map(interactions)
        self._fit_fm(ints_clean, usd_clean, isd_clean)


    def _build_map(self, interactions):
        custs = interactions['User_ID'].unique()
        cid = np.arange(len(custs))

        items = interactions['Style_ID'].unique()
        iid = np.arange(len(items))

        mappings = {}
        mappings['cust_to_id'] = dict(zip(custs, cid))
        mappings['id_to_item'] = dict(zip(iid, items))

        self.mapping = mappings


    def _fit_fm(self, r, u, i):
        ins = self._fm_model(r, u, i)
        solver = aml.SolverFactory('ipopt', keepfiles=False)
        solver.options['max_cpu_time'] = 180
        solver.solve(ins, tee=self.verbose)

        self.rhat = np.zeros(r.shape)
        for c in ins.custs:
            for i in ins.items:
                self.rhat[c,i] = ins.rhat[c,i].value

        self.model = ins


    def _fm_model(self, rating, usd, isd, beta=0.002):

        # general initialization
        num_custs, num_items = rating.shape
        num_emb = self.dim
        num_cfeatures = usd.shape[1]
        num_ifeatures = isd.shape[1]
        np.random.seed(618)
        ecf_init = np.random.rand(num_cfeatures, num_custs, num_emb)
        eif_init = np.random.rand(num_ifeatures, num_items, num_emb)

        # model instance
        m = aml.ConcreteModel()

        # sets
        m.custs = range(num_custs)
        m.items = range(num_items)
        m.embeddings = range(num_emb)
        m.cfeatures = range(num_cfeatures)
        m.ifeatures = range(num_ifeatures)

        # variables
        m.b0 = aml.Var(domain=aml.Reals)
        m.bc = aml.Var(m.custs, domain=aml.Reals)
        m.bi = aml.Var(m.items, domain=aml.Reals)

        def initialize_ecf(m, f, c, k):
            return ecf_init[f,c,k]
        m.ecf = aml.Var(m.cfeatures, m.custs, m.embeddings, domain=aml.Reals, initialize=initialize_ecf)

        def initialize_eif(m, f, i, k):
            return eif_init[f,i,k]
        m.eif = aml.Var(m.ifeatures, m.items, m.embeddings, domain=aml.Reals, initialize=initialize_eif)

        m.ec = aml.Var(m.custs, m.embeddings, domain=aml.Reals)
        m.ei = aml.Var(m.items, m.embeddings, domain=aml.Reals)
        m.rhat = aml.Var(m.custs, m.items, domain=aml.Reals)

        # customer embeddings
        def ec_rule(m, c, k):
            return m.ec[c,k] == sum(m.ecf[f,c,k]*usd[c,f] for f in m.cfeatures)
        m.ec_eqn = aml.Constraint(m.custs, m.embeddings, rule=ec_rule)

        # item embeddings
        def ei_rule(m, i, k):
            return m.ei[i,k] == sum(m.eif[f,i,k]*isd[i,f] for f in m.ifeatures)
        m.ei_eqn = aml.Constraint(m.items, m.embeddings, rule=ei_rule)

        # predicted ratings
        def rhat_rule(m, c, i):
            return m.rhat[c,i] == tanh(m.b0 + m.bc[c] + m.bi[i] + sum(m.ec[c,k] * m.ei[i,k] for k in m.embeddings))
        m.rhat_eqn = aml.Constraint(m.custs, m.items, rule=rhat_rule)

        # objective
        def obj_rule(m):
            e = 0.0
            for c in m.custs:
                for i in m.items:
                    if rating[c,i] > 0 or rating[c,i] < 0:
                        e = e + pow(rating[c,i] - m.rhat[c,i], 2)
                        e = e + (beta/2) * (
                            pow(m.bc[c],2) + pow(m.bi[i],2) + 
                            sum(pow(m.ecf[f,c,k],2) for f in m.cfeatures for k in m.embeddings) + 
                            sum(pow(m.eif[f,i,k],2) for f in m.ifeatures for k in m.embeddings)
                            )
            return e
        m.obj = aml.Objective(rule=obj_rule, sense=aml.minimize)

        return m

