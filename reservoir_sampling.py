"""
https://en.wikipedia.org/wiki/Reservoir_sampling
"""
import numpy as np


SEED = 0
rng = np.random.default_rng(seed=SEED)


class Stream:
    def __init__(self, n: int):
        """
        :param n: length of the data stream, which is assumed to be unknown to the algorithm.
        """
        self._n = n

    def __iter__(self):
        for i in range(self._n):
            yield i


class Reservoir:
    def __init__(self, size: int):
        self._size = size
        self._reservoir = []

    def add(self, i_th, item):
        if i_th <= self._size:
            # when the reservoir is not full, just append new item
            self._reservoir.append(item)
        else:
            # when the reservoir is full and is facing a new item, we randomly generate an int j from [1, i-th]
            # if j falls in the range [1, k], we replace the j-th item in the reservoir with the new item
            # i.e., for the i-th new item, it has the probability of k / i to be included in the reservoir
            if rng.random() <= self._size / i_th:
                # randomly remove an item from the reservoir
                rm_idx = rng.integers(low=0, high=self._size)
                self._reservoir[rm_idx] = item

    @property
    def reservoir(self):
        return self._reservoir


if __name__ == "__main__":
    # data stream length = 50
    N = 50
    # reservoir size
    K = 10
    # we simulate many times to test this algorithm
    N_SIM = 10000

    count = {}
    for _ in range(N_SIM):
        stream = Stream(N)
        reservoir = Reservoir(K)
        for i, item in enumerate(stream):
            i_th = i + 1
            reservoir.add(i_th, item)
        sampled = reservoir.reservoir
        for each_sample in sampled:
            if each_sample not in count:
                count[each_sample] = 1
            else:
                count[each_sample] += 1
    total_counts = K * N_SIM
    probability = {k: v / total_counts for k, v in count.items()}
    for prob in probability.values():
        print(f"\nExpect {K / N}, got {prob}")
