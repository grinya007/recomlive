import numpy as np
import scipy.sparse as sp
import sklearn.preprocessing as pp

from bisect import bisect_left
from cache import Cache, Deque
from collections import namedtuple
from math import log


class Recommender(object):
    def __init__(
            self,
            documents_n      = 2000,
            persons_n        = 2000,
            invalidate_after = 5,
            recs_limit       = 10):

        self.documents_n     = documents_n
        self.persons_n       = persons_n
        self.matrix          = Matrix(documents_n, persons_n)
        self.documents_cache = Cache(documents_n)
        self.persons_cache   = Cache(persons_n)
        self.invalidate_after = invalidate_after
        self.recs_limit      = recs_limit

    def person_history(self, person_id):
        prs_res = self.persons_cache.get_by_key(person_id)
        if prs_res == None:
            return []
        return prs_res.value.history.keys()

    def record(self, document_id, person_id):
        
        new_doc = lambda: Document(document_id, self.invalidate_after, self.recs_limit)
        doc_res = self.documents_cache.get_replace(document_id, new_doc)

        new_prs = lambda: Person(person_id, max(self.documents_n / 10, 10))
        prs_res = self.persons_cache.get_replace(person_id, new_prs)

        if not doc_res.is_hit:
            self.matrix.zero_i(doc_res.idx)
        if not prs_res.is_hit:
            self.matrix.zero_j(prs_res.idx)

        prev_visit = prs_res.value.set_next_visit(document_id)
        if prev_visit != None:
            prev_doc_res = self.documents_cache.get_by_key(prev_visit.document_id)
            if prev_doc_res != None and prev_doc_res.idx != doc_res.idx:
                self._record_dp(doc_res, prs_res, 2)
                if prev_visit.is_first and prev_doc_res.idx != doc_res.idx:
                    self._record_dp(prev_doc_res, prs_res, 1)

    def _record_dp(self, doc, prs, add):
        doc.value.incr_counter()
        total_nz = 2 + self.matrix.nz
        doc_nz = 1 + self.matrix.get_nz_i(doc.idx)
        prs_weight = log(add + self.matrix.get_nz_j(prs.idx))
        rating = log(total_nz/doc_nz)/prs_weight
        self.matrix.set(doc.idx, prs.idx, rating)

    def recommend(self, document_id, person_id = None):
        doc_res = self.documents_cache.get_by_key(document_id)
        if doc_res == None or not self.matrix.has_nz_i(doc_res.idx):
            return []

        if doc_res.value.recs.has_some() and not doc_res.value.must_invalidate():
            return self.filter(doc_res.value.recs.rlist, person_id)

        doc_res.value.restart_counter()
        r = self.matrix.cosine(doc_res.idx)
        if r.size == 0:
            return []

        r = list(enumerate(r.tolist()[0]))
        r.sort(key = lambda r: r[1], reverse = True)
        doc_res.value.recs.wear_out()

        j = 0
        for (i, w) in r:
            if w == 0:
                break

            if i == doc_res.idx:
                continue

            rec_res = self.documents_cache.get_by_idx(i)
            if rec_res == None:
                continue

            doc_res.value.recs.insort([rec_res.key, w])
            j += 1
            if j == self.recs_limit * 10:
                break
        
        return self.filter(doc_res.value.recs.rlist, person_id)

    def filter(self, rlist, person_id):
        if person_id == None:
            return list(map(lambda r: r[0], rlist))
        prs_res = self.persons_cache.get_by_key(person_id)
        if prs_res == None:
            return list(map(lambda r: r[0], rlist))

        j = 0
        frlist = []
        for rec in rlist:
            if not rec[0] in prs_res.value.history:
                frlist.append(rec[0])
                j += 1
                if j == self.recs_limit:
                    break
        return frlist


class Document(object):
    def __init__(self, did, invalidate_after, recs_limit):
        self.id = did
        self.counter = 0
        self.invalidate_after = invalidate_after
        self.recs = Recommendations([], recs_limit)

    def incr_counter(self):
        self.counter += 1
    def restart_counter(self):
        self.counter = 0
    def must_invalidate(self):
        return self.counter >= self.invalidate_after


PrevVisit = namedtuple('PrevVisit', 'document_id, is_first')
class Person(object):
    def __init__(self, pid, history_max_length):
        self.id = pid
        self.history = Deque()
        self.history_max_length = history_max_length

    def set_next_visit(self, document_id):
        if len(self.history) == self.history_max_length:
            self.history.pop()
        if len(self.history) == 0 or self.history[-1][0] != document_id:
            self.history.appendleft(document_id, True)
        if len(self.history) == 1:
            return None
        prev = self.history[-2]
        return PrevVisit(prev[0], len(self.history) == 2)


class Recommendations(object):
    def __init__(self, rlist, limit):
        self.rlist = rlist
        self.limit = limit
        self.rdict = {}
        for (i, rec) in enumerate(self.rlist):
            self.rdict[rec[0]] = i

    def has_some(self):
        return len(self.rlist) > 0

    def wear_out(self):
        for r in self.rlist:
            r[1] *= 0.9

    def insort(self, rec):
        if not self.has_some():
            self.rlist.append(rec)
            self.rdict[rec[0]] = 0
            return

        if rec[0] in self.rdict:
            del self.rlist[self.rdict[rec[0]]]

        weights = list(map(lambda r: -r[1], self.rlist))
        i = bisect_left(weights, -rec[1])
        self.rlist.insert(i, rec)
        self.rdict = dict(map(lambda iv: (iv[1][0], iv[0]), enumerate(self.rlist)))

        if len(self.rlist) > self.limit:
            delrec = self.rlist.pop()
            del self.rdict[delrec[0]]

class Matrix(object):
    def __init__(self, i, j):
        self.m = sp.lil_matrix((i, j))
        self.i_nz = np.zeros(i)
        self.j_nz = np.zeros(j)
        self.nz = 0

    def __repr__(self):
        return str(np.around(self.m.todense(), decimals = 2))

    def set(self, i, j, value):
        if value != 0 and self.m[i, j] == 0:
            self.i_nz[i] += 1
            self.j_nz[j] += 1
            self.nz += 1
        self.m[i, j] = value
        
    def cosine(self, to, axis = 0, normalize = True):
        if to > self.m.shape[axis] - 1:
            raise ValueError
        m = self.m
        if axis == 1:
            m = m.T

        zero_idx_set = set(np.where(m[to].toarray() == 0)[1])
        if len(zero_idx_set) == m.shape[1]:
            return np.array([])

        m = m[:, list(set(range(m.shape[1])) - zero_idx_set)]
        if normalize:
            m = pp.normalize(m)
        return (m * m.T)[to].toarray()

    def has_nz_i(self, i):
        return self.i_nz[i] != 0

    def has_nz_j(self, j):
        return self.j_nz[j] != 0

    def get_nz_i(self, i):
        return self.i_nz[i]

    def get_nz_j(self, j):
        return self.j_nz[j]

    def zero_i(self, i):
        if self.i_nz[i] == 0:
            return
        i_nz = self.m[i].nonzero()[1].tolist()
        self.nz -= len(i_nz)
        self.i_nz[i] = 0
        for j in i_nz:
            if self.m[i, j] != 0:
                self.m[i, j] = 0
                self.j_nz[j] -= 1

    def zero_j(self, j):
        if self.j_nz[j] == 0:
            return
        j_nz = self.m[:, j].nonzero()[0].tolist()
        self.nz -= len(j_nz)
        self.j_nz[j] = 0
        for i in j_nz:
            if self.m[i, j] != 0:
                self.m[i, j] = 0
                self.i_nz[i] -= 1

