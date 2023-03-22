import numpy as np


# 1-D lossless linear scale
def scale(ar: np.ndarray, size: int, axis=-1) -> [np.ndarray, np.ndarray]:
    # swap related axis to top
    if axis != 0:
        ar = np.swapaxes(ar, 0, axis)
    diff = np.diff(ar.astype(float), axis=0)
    p = np.linspace(0, ar.shape[0], size, endpoint=False)
    left_p = p.astype(int)
    delta_x = p - left_p
    delta_y = diff[left_p] * delta_x[:, None, None]
    ret = ar[left_p] + delta_y

    if axis != 0:
        ret, diff = np.swapaxes(ret, 0, axis), np.swapaxes(diff, 0, axis)
    return ret, diff


def inverse_scale(x: np.ndarray, diff: np.ndarray, axis=-1) -> np.ndarray:
    # swap related axis to top
    if axis != 0:
        x = np.swapaxes(x, 0, axis)
        diff = np.swapaxes(diff, 0, axis)
    size = diff.shape[0] + 1
    assert x.shape[0] < size, 'current size must less than original size'
    p = np.linspace(0, size, x.shape[0], endpoint=False)   # restore sample points
    left_p = p.astype(int)
    related_diff = (diff[left_p])
    left, right = x - (p-left_p)[:, None, None] * related_diff, x + (1-p+left_p)[:, None, None] * related_diff
    shape = np.array(diff.shape)
    shape[0] += 1
    ret = np.zeros(shape)
    ret[left_p] = left
    ret[left_p+1] = right

    # restore only when scale greater than 2x
    restored_idx = np.unique(np.hstack((left_p, left_p+1)))
    indices_will_restore_count = 1
    remains = None
    while indices_will_restore_count:
        remains = np.setdiff1d(np.arange(size), restored_idx, assume_unique=True)
        indices_will_restore = remains[:-1][np.diff(remains) > 1]
        ret[indices_will_restore] = ret[indices_will_restore+1] - diff[indices_will_restore]

        restored_idx = np.hstack((restored_idx, indices_will_restore))
        indices_will_restore_count = indices_will_restore.size
    for i in remains:
        ret[i] = ret[i-1] + diff[i-1]

    if axis != 0:
        ret = np.swapaxes(ret, 0, axis)
    return ret


def scale2(ar: np.ndarray, shape) -> [np.ndarray, (np.ndarray, np.ndarray)]:
    shape = np.array(shape)
    ret = ar.copy()
    diff = []
    for axis in np.argwhere(shape != ar.shape):
        axis = axis[0]
        ret, d = scale(ret, shape[axis], axis=axis)
        diff.append(d)
    return ret, diff


def iscale2(ar: np.ndarray, diff) -> np.ndarray:
    for d in diff[::-1]:
        axis = np.argwhere(np.array(ar.shape) != d.shape).flatten()[0]
        ar = inverse_scale(ar, d, axis)
    return ar

