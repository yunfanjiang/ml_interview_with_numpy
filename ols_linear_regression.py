"""
https://en.wikipedia.org/wiki/Simple_linear_regression
"""
import numpy as np
import matplotlib.pyplot as plt


SEED = 0


class OLS:
    def __init__(self):
        self._slope = 0
        self._bias = 0

    @property
    def slope(self):
        return self._slope

    @property
    def bias(self):
        return self._bias

    def fit(self, xs: np.ndarray, ys: np.ndarray):
        """
        slope = \frac{\sum (x - x_mean)(y - y_mean)}{\sum (x - x_mean)^2}
        bias = y_mean - slope * x_mean
        :param xs:
        :param ys:
        :return:
        """
        x_mean = xs.mean()
        y_mean = ys.mean()

        slope = ((xs - x_mean) * (ys - y_mean)).sum() / ((xs - x_mean) ** 2).sum()
        bias = y_mean - slope * x_mean
        self._slope = slope
        self._bias = bias


if __name__ == "__main__":
    rng = np.random.default_rng(seed=SEED)
    # the true model
    true_slope = 5
    true_bias = 4
    xs = np.arange(100)
    true_ys = xs * true_slope + true_bias
    # observations are noisy
    observed_ys = true_ys + rng.uniform(low=-1, high=1, size=true_ys.shape) * 200

    for x, observed_y in zip(xs, observed_ys):
        plt.scatter(x=x, y=observed_y, c="b")
    plt.plot(xs, true_ys, c="g", label="True")

    regressor = OLS()
    regressor.fit(xs=xs, ys=observed_ys)
    fitted_ys = xs * regressor.slope + regressor.bias
    plt.plot(xs, fitted_ys, label="Fitted", c="r")

    plt.legend(loc="best")
    plt.show()
