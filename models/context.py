import numpy as np
from . import hamming_code


class WatermarkContext:
    jpeg_block_size = 8
    freq_band_to_be_embedded = 6
    wm_strength = None
    _wm_bits = None
    wm_bits_restore = None
    wm_bits_index: np.ndarray = None
    wm_size = None
    debug = False
    wavelet_name = 'haar'
    wavelet_component_index = 0  # a h v d
    image = None
    dest_image = None
    image_block_shape = None
    errors = None
    sub_context = None
    bytes_per_second = 2

    block_indices = None    # block indices where the wm bit will be embedded
    block_indices_caused_error_bit = None

    embeder, extractor = None, None
    encoder, decoder = hamming_code.encode, hamming_code.decode

    def __init__(self, wm_bits=None, wm_size=None, wm_strength=4, **kwargs):
        self.wm_size = wm_size
        self.wm_bits = wm_bits
        self.wm_strength = wm_strength
        self.errors = {}
        if 'wm_bits' in kwargs:
            self.wm_bits = kwargs.pop('wm_bits')
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def wm_bits(self):
        return self._wm_bits

    @wm_bits.setter
    def wm_bits(self, wm_bits):
        if isinstance(wm_bits, str):
            wm_bits = wm_bits.encode('utf8')
        if isinstance(wm_bits, bytes):
            wm_bits = np.unpackbits(np.array(list(wm_bits), dtype=np.uint8))
        if isinstance(wm_bits, np.ndarray):
            self.wm_size = len(wm_bits)
        self._wm_bits = wm_bits

    def estimate_video_watermark_duration(self):
        return self.wm_bits.size / (self.bytes_per_second*8)

