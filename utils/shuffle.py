import numpy as np
from hashlib import md5


_UINT32_MAX = 2**32


def get_random_machine(password):
    seed = int.from_bytes(md5(password).digest(), 'little') % _UINT32_MAX
    rand = np.random.RandomState(seed)
    return rand


def shuffle_watermark(wm: bytes, password=b'123456', wm_prefix_bits=24):
    if isinstance(wm, bytes):
        wm = np.unpackbits(np.array(list(wm), dtype=np.uint8))
    rand = get_random_machine(password)
    rand.shuffle(wm[wm_prefix_bits:])
    return wm


def sort_watermark(wm: np.ndarray, password=b'123456', wm_prefix_bits=24):
    rand = get_random_machine(password)
    idx = np.arange(wm.size)
    rand.shuffle(idx[wm_prefix_bits:])
    idx = np.argsort(idx[wm_prefix_bits:])
    wm[wm_prefix_bits:] = wm[wm_prefix_bits:][idx]
    return wm


