import numpy as np
import pandas as pd
from scipy.stats import mode


def get_k_nearest_neighbors(
    input_data: np.ndarray, dataset: np.ndarray, label: np.ndarray, k: int
):
    """
    :param input_data: (B, 4)
    :param dataset: (N, 4)
    :param label: (N)
    :param k: Int
    """
    # expand dim to broadcast
    input_data = input_data[..., np.newaxis]  # (B, 4, 1)
    dataset = np.transpose(dataset)[np.newaxis, ...]  # (1, 4, N)
    # calculate Euclidean distance
    distance = np.linalg.norm(input_data - dataset, axis=1)  # (B, N)
    # sort in an ascending order
    sort_idx = np.argsort(distance, axis=1)  # (B, N)
    # prepare labels
    label = np.repeat(label[np.newaxis, ...], repeats=sort_idx.shape[0], axis=0)
    # sort label
    sorted_label = [
        each_label[each_index] for each_label, each_index in zip(label, sort_idx)
    ]
    sorted_label = np.stack(sorted_label, axis=0)
    # select k-nearest neighbors
    neighbors = label[:, :k]
    return neighbors


def get_predictions(neighbors: np.ndarray):
    """
    :param neighbors: (B, K)
    """
    return mode(neighbors, axis=1)[0]


if __name__ == "__main__":
    K = 3

    ds = pd.read_csv(r"datasets/iris.csv")
    # shuffle
    ds = ds.sample(frac=1).reset_index(drop=True)
    features = ds.iloc[:, :4].to_numpy()
    labels = ds.iloc[:, 4].to_numpy()
    # use last N_test data as test data
    N_test = 10
    train_features = features[:-N_test]
    train_labels = labels[:-N_test]
    test_features = features[-N_test:]
    test_labels = labels[-N_test:]

    # (B,)
    neighbors = get_k_nearest_neighbors(
        input_data=test_features, dataset=train_features, label=train_labels, k=K,
    )
    predictions = get_predictions(neighbors)
    assert len(predictions) == len(test_labels)
    print(f"Accuracy: {(predictions == test_labels).mean()}")
