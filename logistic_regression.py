"""
https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/pdfs/40%20LogisticRegression.pdf
"""

from typing import Optional

import numpy as np
import pandas as pd


SEED = 0


class LogisticRegressor:
    def __init__(self, feature_size: int, lr: float, seed: Optional[int] = None):
        self._feature_size = feature_size
        self._lr = lr
        self._rng = np.random.default_rng(seed=seed)

        self._weight = self._rng.random(size=(feature_size, 1))
        self._bias = np.zeros(shape=(1,))

    @staticmethod
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    def forward(self, input: np.ndarray):
        """
        :param input: (B, feature_size)
        """
        assert input.shape[1] == self._feature_size
        return self.sigmoid(input @ self._weight + self._bias[np.newaxis, ...])

    def fit(self, input: np.ndarray, label: np.ndarray):
        """
        :param input: (B, feature_size)
        :param label: (B, 1)
        """
        prediction = self.forward(input)
        assert prediction.shape == label.shape

        # dL / dw = - X^\intercal (Y - Y^\hat)
        # dL / db = - (Y - Y^\hat)
        y_minus_y_hat = label - prediction  # (B, 1)
        dL_dw = -input.T @ y_minus_y_hat  # (feature_size, 1)
        dL_db = -y_minus_y_hat.mean(axis=0)  # (1,)

        # update with gradient descent
        self._weight -= self._lr * dL_dw
        self._bias -= self._lr * dL_db

    def predict(self, input):
        """
        :param input: (B, feature_size)
        :return: (B, 1)
        """
        prob = self.forward(input)
        return np.float32(prob >= 0.5)


if __name__ == "__main__":
    N_FEATURES = 60
    N_EXAMPLES = 208
    N_TRAIN = int(0.7 * N_EXAMPLES)

    names = [f"feature_{i}" for i in range(N_FEATURES)]
    names.append("label")
    ds = pd.read_csv(r"datasets/sonar_all.csv", names=names)
    # convert to numerical label
    ds["label"] = ds["label"].apply(lambda x: 1 if x == "M" else 0)
    # shuffle the dataset
    ds = ds.sample(frac=1, random_state=SEED).reset_index(drop=True)
    train_set = ds[:N_TRAIN].to_numpy()
    test_set = ds[N_TRAIN:].to_numpy()

    train_feature, train_label = train_set[:, :N_FEATURES], train_set[:, N_FEATURES:]
    test_feature, test_label = test_set[:, :N_FEATURES], test_set[:, N_FEATURES:]

    LR = 1e-3
    N_ITERATIONS = 500

    logistic_regressor = LogisticRegressor(feature_size=N_FEATURES, lr=LR, seed=SEED)
    for _ in range(N_ITERATIONS):
        logistic_regressor.fit(train_feature, train_label)
    test_prediction = logistic_regressor.predict(test_feature)
    accuracy = (test_prediction == test_label).mean()
    print(f"\nAccuracy: {accuracy}")
