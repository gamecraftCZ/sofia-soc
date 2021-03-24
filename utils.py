import numpy as np


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def baseline_mean_squared_error_from_rides(rides_flat):
    output_steer = rides_flat[:, 1]

    steer_mean = np.mean(output_steer)

    error = (output_steer - steer_mean)
    error_squared = error ** 2
    mean_squared_error = np.mean(error_squared)

    return mean_squared_error
