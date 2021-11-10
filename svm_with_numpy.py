import numpy as np

LR = 1e-3
LAMBDA = 1e-4
N_STEPS = 1000
IN_SIZE = 3
OUT_SIZE = 1

# input data (IN_SIZE, N)
x = np.array([[-2, 4, -1], [4, 1, -1], [1, 6, -1], [2, 4, -1], [6, 2, -1],])
x = np.transpose(x)

# targets (OUT_SIZE, N)
target = np.array([-1, -1, 1, 1, 1])[np.newaxis, ...]

W = np.random.randn(OUT_SIZE, IN_SIZE)


if __name__ == "__main__":
    for i in range(N_STEPS):
        predictions = np.dot(W, x)  # (OUT_SIZE, N)

        # we use Hinge loss here
        # L = (1 - y * \hat{y})_+ + \lambda \Vert w \Vert^2
        mask = (1 - predictions * target) > 0  # (OUT_SIZE, N)
        hinge_gradient = -np.dot(target * mask, np.transpose(x))  # (OUT_SIZE, IN_SIZE)
        regularization_gradient = 2 * W  # (OUT_SIZE, IN_SIZE)
        total_gradient = hinge_gradient + LAMBDA * regularization_gradient

        # do one step of gradient descent
        W = W - LR * total_gradient

    # now test on some new points
    x_test = np.array([[-1, 3, -1], [5, 5, -1],])
    x_test = np.transpose(x_test)
    predictions_test = np.dot(W, x_test)
    print(f"Targets test: [negative, positive]\n Predicted test: {predictions_test}")
    print(f"Learned W: {W}")
