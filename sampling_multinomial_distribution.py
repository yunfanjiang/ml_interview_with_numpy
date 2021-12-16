from typing import List, Any, Optional

import numpy as np


class MultinomialDistributionSampler:
    def __init__(
        self, weights: List[float], categories: List[Any], seed: Optional[int] = None
    ):
        assert sum(weights) == 1
        self._categories = categories
        self._rng = np.random.default_rng(seed=seed)
        self._cdf = [sum(weights[: i + 1]) for i in range(len(weights))]

    def sample(self):
        random_number = self._rng.random()
        for idx, mass in enumerate(self._cdf):
            if mass >= random_number:
                return self._categories[idx]


if __name__ == "__main__":
    seed = 0
    weights = [0.2, 0.1, 0.3, 0.1, 0.3]
    categories = ["a", "b", "c", "d", "e"]
    sampler = MultinomialDistributionSampler(
        weights=weights, categories=categories, seed=seed
    )

    N_SIM = 100000
    counts = {k: 0 for k in categories}
    for _ in range(N_SIM):
        counts[sampler.sample()] += 1

    frequencies = {k: v / N_SIM for k, v in counts.items()}
    for i, freq in enumerate(frequencies.values()):
        print(f"\n Expect {weights[i]}, got {freq}")
