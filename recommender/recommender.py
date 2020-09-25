import numpy as np

from cache import Cache, Deque
from collections import namedtuple
from .rnn import RNN
from math import log


class Recommender(object):
    def __init__(
            self,
            documents_n      = 2000,
            persons_n        = 2000,
            recs_limit       = 10,
            device           = 'cuda'
        ):

        self.documents_n     = documents_n
        self.persons_n       = persons_n
        self.documents_cache = Cache(documents_n)
        self.persons_cache   = Cache(persons_n)
        self.recs_limit      = recs_limit
        self.lstm            = RNN(documents_n, 320, 128, device)

    def person_history(self, person_id):
        prs_res = self.persons_cache.get_by_key(person_id)
        if prs_res == None:
            return []
        return prs_res.value.history.keys()

    def record(self, document_id, person_id):
        
        doc_res = self.documents_cache.get_replace(document_id, None)

        new_prs = lambda: Person(person_id, max(self.documents_n / 10, 10))
        prs_res = self.persons_cache.get_replace(person_id, new_prs)

        prs_res.value.append_history(document_id)
        unlearned = prs_res.value.unlearned_docs()
        if len(unlearned) == 2:
            inputs = []
            for document_id in reversed(unlearned):
                doc_res = self.documents_cache.get_by_key(document_id)
                if doc_res != None:
                    inputs.append(doc_res.idx)
            if len(inputs) >= 2:
                loss = self.lstm.fit(inputs)
                prs_res.value.mark_learned(unlearned)


    def recommend(self, document_id, person_id = None):
        doc_res = self.documents_cache.get_by_key(document_id)
        if doc_res == None:
            return []

        history = {}
        if person_id != None:
            prs_res = self.persons_cache.get_by_key(person_id)
            if prs_res != None:
                history = prs_res.value.history

        r = self.lstm.predict(doc_res.idx)

        recs = []
        for i in r:
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



class Person(object):
    def __init__(self, pid, history_max_length):
        self.id = pid
        self.history = Deque()
        self.history_max_length = history_max_length
        self.lstm_h = None

    def append_history(self, document_id):
        if len(self.history) == self.history_max_length:
            self.history.pop()
        if len(self.history) == 0 or self.history[-1][0] != document_id:
            self.history.appendleft(document_id, False)
    
    def unlearned_docs(self):
        doc_ids = []
        for doc_id in self.history:
            if self.history.od[doc_id]:
                break
            doc_ids.append(doc_id)
        return doc_ids

    def mark_learned(self, doc_ids):
        for doc_id in doc_ids[1:]:
            if doc_id in self.history:
                self.history.od[doc_id] = True


