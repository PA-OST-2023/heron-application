import numpy as np

def peak_detector(x):
    pass

def arg_max_detector(x):
    return [np.unravel_index(np.argmax(x), x.shape)]
