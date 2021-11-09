import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def loss(y, o):
    return (0.5 * np.square(y - o)).mean()


def d_loss(y, o):
    return o - y  # (OUT_SIZE, N)


BATCH_SIZE = 32
LR = 5e-4

INPUT_SIZE = 1
OUTPUT_SIZE = 1
HIDDEN_SIZE = 32


# initialize weights
W1 = np.random.randn(HIDDEN_SIZE, INPUT_SIZE)
b1 = np.zeros(shape=(HIDDEN_SIZE, 1), dtype=np.float32)
W2 = np.random.randn(OUTPUT_SIZE, HIDDEN_SIZE)
b2 = np.zeros(shape=(OUTPUT_SIZE, 1), dtype=np.float32)


# forward pass
def forward(x):
    """
    x: (1 , N)
    """
    z1 = np.dot(W1, x) + b1  # (HIDDEN, N)
    a1 = sigmoid(z1)  # (HIDDEN, N)
    o = np.dot(W2, a1) + b2  # (OUT, N)
    return o, z1, a1


# back propagation
def backward(x, z1, a1, o, y, w2):
    dl_do = d_loss(y=y, o=o)  # (OUT_SIZE, N)
    dL_dW2 = np.dot(dl_do, np.transpose(a1))  # (OUT_SIZE, HIDDEN)
    dL_db2 = np.mean(dl_do, axis=-1, keepdims=True)  # (OUT_SIZE, 1)

    # now we need to calculate dL_dz, which is of shape (HIDDEN, N)
    # to compute dL_dz, we need to compute dL_da, which is of shape (HIDDEN, N)
    dL_da = np.dot(np.transpose(w2), dl_do)  # (HIDDEN, N)
    dL_dz = dL_da * d_sigmoid(z1)  # (HIDDEN, N)
    dL_dW1 = np.dot(dL_dz, np.transpose(x))  # (HIDDEN, IN)
    dL_db1 = np.mean(dL_dz, axis=-1, keepdims=True)  # (OUT_SIZE, 1)
    return {
        "dL_dW2": dL_dW2,
        "dL_db2": dL_db2,
        "dL_dW1": dL_dW1,
        "dL_db1": dL_db1,
    }


if __name__ == "__main__":
    # data generation fn
    dg_fn = np.cos

    EPSILON = 5e-4

    i = 0
    while True:
        # sample a batch of data
        inputs = np.random.rand(1, BATCH_SIZE) * 2 * np.pi - np.pi  # (1, N) [-pi, +pi]
        targets = dg_fn(inputs)  # (1, N)

        # forward the NN
        predicts, z, a = forward(inputs)

        # calculate loss
        loss_value = loss(targets, predicts)
        print(f"{i}-iteration, loss: {loss_value}\n")
        if loss_value <= EPSILON:
            break

        # back propagate
        gradients = backward(x=inputs, z1=z, a1=a, o=predicts, y=targets, w2=W2,)

        # now update parameters
        W1 -= LR * gradients["dL_dW1"]
        b1 -= LR * gradients["dL_db1"]
        W2 -= LR * gradients["dL_dW2"]
        b2 -= LR * gradients["dL_db2"]

        i += 1

    # after training, test some special cases
    tests_points = np.array([[0], [np.pi / 6], [np.pi / 3], [np.pi / 2]])
    tests_points = np.transpose(tests_points)
    predicts = forward(tests_points)[0]
    print(predicts)
