from collections import OrderedDict, namedtuple
from copy import copy

IDX = 0
KEY = 1
VAL = 2

Result = namedtuple('Result', 'is_hit, idx, key, value')
class Cache(object):
    def __init__(self, size):
        self.size = size
        self.t1_size = 0

        self.key_map = {}
        self.idx_pool = list(map(lambda i: [i, None, None], range(size)))
        self.idx_map = copy(self.idx_pool)

        self.t1 = Deque()
        self.b1 = Deque()

        self.t2 = Deque()
        self.b2 = Deque()

    def __repr__(self):
        t1 = list(map(lambda k: (self.key_map[k][IDX], k), self.t1))
        t2 = list(map(lambda k: (self.key_map[k][IDX], k), self.t2))
        return 'T1: {}\nT2: {}'.format(t1, t2)

    def get_by_key(self, key):
        val = self.key_map.get(key)
        if val == None:
            return None
        return Result(True, val[IDX], val[KEY], val[VAL])

    def get_by_idx(self, idx):
        if isinstance(idx, int) and idx >= 0 and idx < len(self.idx_map):
            val = self.idx_map[idx]
            if val[KEY] == None:
                return None
            return Result(True, val[IDX], val[KEY], val[VAL])
        else:
            return None

    def get_replace(self, key, onmiss = None):
        if key in self.key_map:
            if key in self.t1:
                self.t1.remove(key)
                self.t2.appendleft(key)
            else:
                self.t2.remove(key)
                self.t2.appendleft(key)
            hit_val = self.key_map[key]
            return Result(True, hit_val[IDX], hit_val[KEY], hit_val[VAL])

        val = onmiss() if hasattr(onmiss, '__call__') else onmiss

        if key in self.b1 or key in self.b2:
            if key in self.b1:
                self.t1_size = min(self.size, self.t1_size + max(len(self.b2) / len(self.b1), 1))
                self.adjust(key)
                self.b1.remove(key)
            else:
                self.t1_size = max(0, self.t1_size - max(len(self.b1) / len(self.b2), 1))
                self.adjust(key)
                self.b2.remove(key)
            miss_val = self.idx_pool.pop()
            miss_val[KEY] = key
            miss_val[VAL] = val
            self.key_map[key] = miss_val
            self.t2.appendleft(key)
            return Result(False, miss_val[IDX], miss_val[KEY], miss_val[VAL])

        if len(self.t1) + len(self.b1) == self.size:
            if len(self.t1) < self.size:
                self.b1.pop()
                self.adjust(key)
                miss_val = self.idx_pool.pop()
                miss_val[KEY] = key
                miss_val[VAL] = val
                self.key_map[key] = miss_val
            else:
                evict = self.t1.pop()
                evict_val = self.key_map[evict]
                evict_val[KEY] = None
                evict_val[VAL] = None
                self.idx_pool.append(evict_val)
                del self.key_map[evict]

        else:
            total = len(self.t1) + len(self.b1) + len(self.t2) + len(self.b2)
            if total >= self.size:
                if total == (2 * self.size):
                    self.b2.pop()

                self.adjust(key)
                miss_val = self.idx_pool.pop()
                miss_val[KEY] = key
                miss_val[VAL] = val
                self.key_map[key] = miss_val

        self.t1.appendleft(key)
        if not key in self.key_map:
            miss_val = self.idx_pool.pop()
            miss_val[KEY] = key
            miss_val[VAL] = val
            self.key_map[key] = miss_val

        miss_val = self.key_map[key]
        return Result(False, miss_val[IDX], miss_val[KEY], miss_val[VAL])

    def adjust(self, key):
        if self.t1 and ((key in self.b2 and len(self.t1) == self.t1_size) or (len(self.t1) > self.t1_size)):
            evict = self.t1.pop()
            self.b1.appendleft(evict)
        else:
            evict = self.t2.pop()
            self.b2.appendleft(evict)

        evict_val = self.key_map[evict]
        evict_val[KEY] = None
        evict_val[VAL] = None
        self.idx_pool.append(evict_val)
        del self.key_map[evict]


class Deque(object):
    def __init__(self):
        self.od = OrderedDict()

    def appendleft(self, k, v = None):
        if k in self.od:
            del self.od[k]
        self.od[k] = v

    def pop(self):
        return self.od.popitem(0)[0]

    def remove(self, k):
        del self.od[k]

    def keys(self):
        return list(self.od)

    def __len__(self):
        return len(self.od)

    def __contains__(self, k):
        return k in self.od

    def __getitem__(self, i):
        return list(self.od.items())[i]

    def __iter__(self):
        return reversed(self.od)


