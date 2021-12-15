"""
Adapted from https://github.com/random-forests/tutorials/blob/master/decision_tree.py
"""


from typing import Union, List, Tuple


# data structure for a training point/test point
# represents feature color, feature size, and label, respectively
DATA = Tuple[str, float, str]
BATCH_DATA = List[DATA]


FEATURES = ["color", "size"]
# column 0: feature color, column 1: feature size, column 2: label
TRAIN_DATA = [
    ("Green", 3.0, "Apple"),
    ("Yellow", 3.0, "Apple"),
    ("Red", 1.0, "Grape"),
    ("Red", 1.0, "Grape"),
    ("Yellow", 3.0, "Lemon"),
]


class Question:
    """
    Split based on  the `feature_column` feature with decision value `decision_value`
    """

    def __init__(self, feature_column: int, decision_value: Union[str, float]):
        self._feature_column = feature_column
        self._decision_value = decision_value

    def match(self, query: DATA):
        query_feature = query[self._feature_column]
        if isinstance(query_feature, str):
            return query_feature == self._decision_value
        elif isinstance(query_feature, float):
            return query_feature >= self._decision_value
        else:
            raise ValueError(f"Unknown type {type(query_feature)}")

    def __repr__(self):
        # This is just a helper method to print
        # the question in a readable format.
        condition = ">="
        if isinstance(self._decision_value, str):
            condition = "=="
        return "Is %s %s %s?" % (
            FEATURES[self._feature_column],
            condition,
            str(self._decision_value),
        )


def partition(batch_query_data: BATCH_DATA, question: Question):
    true_list, false_list = [], []
    for query_data in batch_query_data:
        if question.match(query_data):
            true_list.append(query_data)
        else:
            false_list.append(query_data)
    return true_list, false_list


def gini(batch_data: BATCH_DATA):
    unique_labels = set([data[-1] for data in batch_data])
    unique_labels_counts = {label: 0 for label in unique_labels}
    for label in unique_labels:
        for data in batch_data:
            if data[-1] == label:
                unique_labels_counts[label] += 1
    n_data = len(batch_data)
    gini_impurity = 1.0
    for count in unique_labels_counts.values():
        gini_impurity -= (count / n_data) ** 2
    return gini_impurity


def information_gain(
    true_branch: BATCH_DATA, false_branch: BATCH_DATA, prev_gini: float
):
    n_true, n_false = len(true_branch), len(false_branch)
    true_gini, false_gini = gini(true_branch), gini(false_branch)
    post_gini = (n_true / (n_true + n_false)) * true_gini + (
        n_false / (n_true + n_false)
    ) * false_gini
    return prev_gini - post_gini


def find_best_split(batch_data: BATCH_DATA) -> (float, Question):
    """
    Take a batch of data, find the best question that splits those data and return the corresponding information gain
    """
    prev_gini = gini(batch_data)

    best_info_gain, best_question = 0, None
    num_features = len(batch_data[0]) - 1  # the last is label

    # maintain a history of evaluated feature, we don't need to evaluate multiple times
    evaluation_history = {idx: [] for idx in range(num_features)}

    for each_data in batch_data:
        for feature_idx in range(num_features):
            if each_data[feature_idx] in evaluation_history[feature_idx]:
                continue
            else:
                candidate_question = Question(feature_idx, each_data[feature_idx])
                true_list, false_list = partition(batch_data, candidate_question)
                info_gain = information_gain(true_list, false_list, prev_gini)
                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    best_question = candidate_question
                evaluation_history[feature_idx].append(each_data[feature_idx])
    return best_info_gain, best_question


# structure for decision node
class Node:
    pass


class DecisionNode(Node):
    def __init__(self, question: Question, true_branch: Node, false_branch: Node):
        self._question = question
        self._true_branch = true_branch
        self._false_branch = false_branch

    @property
    def question(self):
        return self._question

    @property
    def true_branch(self):
        return self._true_branch

    @property
    def false_branch(self):
        return self._false_branch


# leaf node, make predictions
class Leaf(Node):
    def __init__(self, batch_data: BATCH_DATA):
        self._batch_data = batch_data

    def predict(self):
        # return the label distribution of the batch data
        unique_labels = set([data[-1] for data in self._batch_data])
        unique_labels_counts = {label: 0 for label in unique_labels}
        for label in unique_labels:
            for data in self._batch_data:
                if data[-1] == label:
                    unique_labels_counts[label] += 1
        prediction = {
            label: count / sum(unique_labels_counts.values())
            for label, count in unique_labels_counts.items()
        }
        return prediction


# recursively learn a tree
def build_a_tree(batch_data: BATCH_DATA):
    # find the partition that best splits the data
    info_gain, question = find_best_split(batch_data)

    if info_gain == 0:
        # if no gain, we reach a leaf
        return Leaf(batch_data)

    # now we use the previously found best question to split our data
    true_list, false_list = partition(batch_data, question)

    # now expand on those two branches
    true_node = build_a_tree(true_list)
    false_node = build_a_tree(false_list)
    return DecisionNode(question, true_node, false_node)


def print_tree(node, spacing=""):
    """World's most elegant tree printing function."""

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        print(spacing + "Predict", node.predict())
        return

    # Print the question at this node
    print(spacing + str(node.question))

    # Call this function recursively on the true branch
    print(spacing + "--> True:")
    print_tree(node.true_branch, spacing + "  ")

    # Call this function recursively on the false branch
    print(spacing + "--> False:")
    print_tree(node.false_branch, spacing + "  ")


def predict(node: Union[DecisionNode, Leaf, Node], data: DATA):
    """
    recursively predict
    """
    if isinstance(node, Leaf):
        return node.predict()
    elif isinstance(node, DecisionNode):
        if node.question.match(data):
            return predict(node.true_branch, data)
        else:
            return predict(node.false_branch, data)


if __name__ == "__main__":
    tree = build_a_tree(TRAIN_DATA)
    print_tree(tree)

    # labels are masked
    TEST_DATA = [
        ("Green", 3.0, None),
        ("Yellow", 4.0, None),
        ("Red", 2.0, None),
        ("Red", 1.0, None),
        ("Yellow", 3.0, None),
    ]
    TEST_LABELS = ["Apple", "Apple", "Grape", "Grape", "Lemon"]

    predictions = []
    for data in TEST_DATA:
        predictions.append(predict(tree, data))

    for test_label, prediction in zip(TEST_LABELS, predictions):
        print(f"\nGround truth: {test_label}, Prediction: {prediction}")
