import numpy as np
import pytest
from utils.shuffle import shuffle_watermark, sort_watermark


def test_shuffle_watermark():
    data = b'hello world'
    pwd = b'cjh'
    prefix_bytes = 3
    pkt = shuffle_watermark(data, pwd, wm_prefix_bits=8*prefix_bytes)
    assert np.packbits(pkt).tobytes()[:prefix_bytes] == data[:prefix_bytes]
    ret = sort_watermark(pkt, pwd)
    if isinstance(ret, np.ndarray):
        ret = np.packbits(ret).tobytes()
    assert data == ret


