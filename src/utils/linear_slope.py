import numpy as np

def linear_slope(values):
    if len(values) < 2:
        return 0.0

    x = np.arange(len(values))
    slope, _ = np.polyfit(x, values, 1)

    return float(slope)