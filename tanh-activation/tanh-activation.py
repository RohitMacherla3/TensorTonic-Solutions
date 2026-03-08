import numpy as np

def tanh(x):
    """
    Implement Tanh activation function.
    """
    x_np = np.asarray(x)
    exp_x = np.exp(x_np)
    exp_neg_x = np.exp(-x_np)

    return (exp_x - exp_neg_x)/(exp_x + exp_neg_x)