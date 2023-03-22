import numpy as np
import pytest
from models.hamming_code import get_encoded_data_index, encode, decode


def test_hamming_code():
    data = b'hello world'
    pwd = b'cjh'

    pkt = encode(data, shuffle_seed=pwd)
    pkt[9] = not pkt[9]
    pkt[8] = not pkt[8]
    pkt[7] = not pkt[7]
    pkt[6] = not pkt[6]
    pkt[5] = not pkt[5]
    pkt[4] = not pkt[4]
    ret = decode(pkt, shuffle_seed=pwd)
    assert ret == data


def test_get_encoded_data_index():
    data = b'hello world'
    pwd = b'cjh'

    pkt = encode(data, shuffle_seed=pwd)
    idx = get_encoded_data_index(len(data)*8, shuffle_seed=pwd)[:24]
    assert np.packbits(pkt[idx]).tobytes() == data[:3]

