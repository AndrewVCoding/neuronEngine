import math


def step_linear(x):
    """
    0 when x < 0
    1 when x = 0
    x when x > 0

    f(x) = log(1 + e^(x-3)) + (1/(1+e^(-100x + 3)))
    :param x:
    :return:
    """
    try:
        return math.log(1 + math.exp(x - 4)) + (1 / (1 + math.exp(-100 * x + 4)))
    except OverflowError:
        return float('inf')


def linear(x):
    """
    f(x) = x
    :param x:
    :return:
    """
    return x


def relu(x):
    """
    f(x) = log(1 + e^(x-3))
    :param x:
    :return:
    """
    return math.log(1 + math.exp(x-3))


def sigmoid(x):
    """
    f(x) = 1 / (1 + e^(-x))
    :param x:
    :return:
    """
    return 1 / (1 + math.exp(-x))


functions = {'STEP_LINEAR': step_linear, 'LINEAR': linear, 'RELU': relu, 'SIGMOID': sigmoid}


def function(s):
    return functions[s]
