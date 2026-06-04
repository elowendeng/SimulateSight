# utils/lru_cache.py

from collections import OrderedDict


class LRUCache:
    """LRU Cache Implementation. Avoiding Memory Leaks"""

    def __init__(self, maxsize=128):
        self.maxsize = maxsize
        self.cache = OrderedDict()

    def get(self, key):
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.maxsize:
            self.cache.popitem(last=False)

    def clear(self):
        self.cache.clear()