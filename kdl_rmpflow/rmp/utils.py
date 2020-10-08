import numpy as np


def slerp_quat_norm(q1, q2):
    return min(np.arccos(np.sum(q1 * q2)), np.arccos(np.sum(q1 * -q2)))
