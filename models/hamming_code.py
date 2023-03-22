import numpy as np
import hashlib
from hashlib import md5

"""
Refer:
* Hamming 7,4 Explain   https://michaeldipperstein.github.io/hamming.html
* 3Blue1Brown           https://www.youtube.com/watch?v=b3NxrZOu_CE&t=772s
"""


def encode(data, shuffle_seed=None):
    if isinstance(data, bytes):
        data = np.unpackbits(np.array(list(data), dtype=np.uint8))
    data = data.reshape((4, -1))
    ret = np.zeros((7, data.size//4), dtype=np.uint8)
    ret[0] = data[0] ^ data[1] ^ data[3]
    ret[1] = data[0] ^ data[2] ^ data[3]
    ret[2] = data[0]
    ret[3] = data[1] ^ data[2] ^ data[3]
    ret[4:] = data[1:]
    ret = ret.flatten()

    # shuffle bits
    if shuffle_seed is not None:
        seed = int.from_bytes(md5(shuffle_seed).digest(), 'little') % 2**32
        rand = np.random.RandomState(seed)
        rand.shuffle(ret)
    return ret


def decode(data, output_format=bytes, shuffle_seed=None):
    if shuffle_seed:
        idx = np.arange(data.size)
        seed = int.from_bytes(md5(shuffle_seed).digest(), 'little') % 2**32
        rand = np.random.RandomState(seed)
        rand.shuffle(idx)
        data = data[np.argsort(idx)]
    data = data.reshape((7, data.size//7))
    # fix error bit
    err_row = np.zeros(data.shape[1], dtype=np.uint8)
    for i in range(data.shape[0]):
        err_row[np.argwhere(data[i] == 1)] ^= i+1

    err_col = np.argwhere(err_row > 0)
    err_row = err_row[err_col] - 1
    for i in range(err_row.size):
        r, c = err_row[i], err_col[i]
        data[r, c] = not data[r, c]

    data = data[[2, 4, 5, 6]].flatten()
    if output_format == bytes:
        data = np.packbits(data).tobytes()
    return data


def get_encoded_data_index(src_data_bit_size, shuffle_seed=None):
    """get index of embedded data in hamming codes, the returned idx is sorted"""
    data = np.arange(src_data_bit_size).reshape((4, -1))
    ret = np.ones((7, data.size//4), dtype=int) * -1
    ret[2] = data[0]
    ret[4:] = data[1:]
    ret = ret.flatten()

    # shuffle bits
    if shuffle_seed is not None:
        seed = int.from_bytes(md5(shuffle_seed).digest(), 'little') % 2**32
        rand = np.random.RandomState(seed)
        rand.shuffle(ret)

    idx = np.argwhere(ret >= 0).flatten()
    idx = np.array(sorted(idx, key=lambda i: ret[i]))
    return idx
