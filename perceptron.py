import numpy as np
import pandas as pd

SEED = 0


class Perceptron:
    def __init__(self, in_size: int, lr: float):
        rng = np.random.default_rng(seed=SEED)
        self._weight = rng.random((in_size, 1))
        self._bias = np.zeros((1, 1))
        self._lr = lr

    def forward(self, input: np.ndarray):
        """
        :param input: (B, D)
        :return: (B, 1)
        """
        z = input @ self._weight + self._bias  # (B, 1)
        return np.float32(z >= 0)

    def fit(self, input: np.ndarray, labels: np.ndarray):
        """
        :param input: (B, D)
        :param labels: (B, 1)
        """
        predictions = self.forward(input)
        assert predictions.shape == labels.shape

        # calculate delta w & delta b
        # delta w = \alpha * (y - y^\hat) \times x
        errors = labels - predictions  # (B, 1)
        delta_w = self._lr * (np.transpose(input) @ errors)  # (D, 1)
        delta_b = delta_w.copy().mean(axis=0, keepdims=True)  # (1, 1)
        # update weights
        # w' = w + \delta w
        # b' = b + \delta b
        self._weight += delta_w
        self._bias += delta_b
        return errors.mean()

    def predict(self, input: np.ndarray):
        return self.forward(input)


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
    N_ITERATIONS = 100
    perceptron = Perceptron(in_size=N_FEATURES, lr=LR)
    for _ in range(N_ITERATIONS):
        error = perceptron.fit(input=train_feature, labels=train_label)

    test_predictions = perceptron.predict(test_feature)
    accuracy = (test_predictions == test_label).mean()
    print(f"\n Test Accuracy: {accuracy}")
