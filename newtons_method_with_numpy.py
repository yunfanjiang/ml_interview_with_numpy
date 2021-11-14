import numpy as np


# define a quadratic function we aim to solve
def f(x):
    # solutions for f(x) = 0 are 1 and 4
    return (x - 1) * (x - 4)


# now define its derivative
def df(x):
    return 2 * x - 5


def main():
    # define a stop threshold
    eps = 1e-5
    # number of iterations
    n_iterations = 1000

    # random select a candidate solution x_0 between -10 and 10
    x_0 = np.random.uniform(low=-10, high=10)
    # now start iterations
    for i in range(n_iterations):
        # calculate the x intercept of the tangent line at x_0
        # the tangent line is given by y = f'(x_0)(x - x_0) + f(x_0)
        # the x intercept if given by solving y = 0, which gives x = x_0 - f(x_0) / f'(x_0)
        x_intercept = x_0 - f(x_0) / df(x_0)

        if abs(x_intercept - x_0) <= eps:
            print(f"Converged in iteration {i}, solution: {x_intercept}")
            return
        x_0 = x_intercept
    print(f"Not converged within {n_iterations} iterations. Final result: {x_0}")


if __name__ == "__main__":
    main()
