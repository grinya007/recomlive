import numpy as np

from cache import Cache, Deque
from collections import namedtuple
from .lstm import LSTM
from math import log


class Recommender(object):
    def __init__(
            self,
            documents_n      = 2000,
            persons_n        = 2000,
            recs_limit       = 10):

        self.documents_n     = documents_n
        self.persons_n       = persons_n
        self.documents_cache = Cache(documents_n)
        self.persons_cache   = Cache(persons_n)
        self.recs_limit      = recs_limit
        self.lstm            = LSTM(documents_n, int(documents_n * 0.15))
        self.losses          = []
        self.losses_length   = 50

    def person_history(self, person_id):
        prs_res = self.persons_cache.get_by_key(person_id)
        if prs_res == None:
            return []
        return prs_res.value.history.keys()

    def record(self, document_id, person_id):
        
        new_doc = lambda: Document(document_id)
        doc_res = self.documents_cache.get_replace(document_id, new_doc)

        new_prs = lambda: Person(person_id, max(self.documents_n / 10, 10))
        prs_res = self.persons_cache.get_replace(person_id, new_prs)

        prev_visit = prs_res.value.set_next_visit(document_id)
        if prev_visit != None:
            prev_doc_res = self.documents_cache.get_by_key(prev_visit.document_id)
            if prev_doc_res != None and prev_doc_res.idx != doc_res.idx:
                inputs = np.zeros((2, self.documents_n))
                inputs[0][prev_doc_res.idx] = 1
                inputs[1][doc_res.idx] = 1
                loss, prs_res.value.lstm_h, prs_res.value.lstm_C = self.lstm.fit(
                    inputs, prs_res.value.lstm_h, prs_res.value.lstm_C, lr = 0.25/(1 + log(len(prs_res.value.history)))
                )
                print(loss, flush=True)

    def recommend(self, document_id, person_id = None):
        doc_res = self.documents_cache.get_by_key(document_id)
        if doc_res == None:
            return []

        lstm_h = None
        lstm_C = None
        history = {}
        if person_id != None:
            prs_res = self.persons_cache.get_by_key(person_id)
            if prs_res != None:
                lstm_h = prs_res.value.lstm_h
                lstm_C = prs_res.value.lstm_C
                history = prs_res.value.history

        X = np.zeros(self.documents_n)
        X[doc_res.idx] = 1
        r, _, _ = self.lstm.predict(X, lstm_h, lstm_C)

        r = list(enumerate(r.T[0].tolist()))
        r.sort(key = lambda r: r[1], reverse = True)

        recs = []
        for (i, w) in r:
            if i == doc_res.idx:
                continue

            rec_res = self.documents_cache.get_by_idx(i)
            if rec_res == None:
                continue

            if rec_res.key in history:
                continue
            
            recs.append(rec_res.key)
            if len(recs) == self.recs_limit:
                break
        
        return recs


class Document(object):
    def __init__(self, did):
        self.id = did


PrevVisit = namedtuple('PrevVisit', 'document_id, is_first')
class Person(object):
    def __init__(self, pid, history_max_length):
        self.id = pid
        self.history = Deque()
        self.history_max_length = history_max_length
        self.lstm_h = None
        self.lstm_C = None

    def set_next_visit(self, document_id):
        if len(self.history) == self.history_max_length:
            self.history.pop()
        if len(self.history) == 0 or self.history[-1][0] != document_id:
            self.history.appendleft(document_id, True)
        if len(self.history) == 1:
            return None
        prev = self.history[-2]
        return PrevVisit(prev[0], len(self.history) == 2)


