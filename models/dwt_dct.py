import numpy as np
import cv2 as cv
import ujson as json
import pywt
from scipy import fftpack
from .base import AbstractWatermark, ImageMeasure
from .arnold import ArnoldTransform

_default_block_size=8


def gaussian(x0, y0=None, sigma_x=1., sigma_y=1., A=1.):
    # https://en.wikipedia.org/wiki/Gaussian_function#Two-dimensional_Gaussian_function
    if y0 is not None:
        sX, sY = 2*sigma_x, 2*sigma_y
        def func(x, y):
            power = ((x-x0)/sX)**2 + ((y-y0)/sY)**2
            val = A * np.exp(-power)
            return val
        f = func
    else:
        sX = 2*sigma_x
        def func(x):
            power = ((x-x0)/sX)**2
            val = A * np.exp(-power)
            return val
        f = func
    return np.vectorize(f)


def _calc_uint8_precision_loss(blocks: np.ndarray, debug=False):
    assert blocks.dtype == float
    errors = {}
    # calc precision loss
    base_u8 = np.round(blocks)
    under, over = base_u8[base_u8 < 0], base_u8[base_u8 > 255]
    errors['over'], errors['under'] =\
        (over.size, over.mean()) if len(over) else None, (under.size, under.mean()) if len(over) else None
    if debug:
        print('PrecisionLoss(over,under):', errors['PrecisionLoss(over,under)'])
    # # get precision loss index
    # under_idx = np.argwhere((base_u8 < 0).sum(axis=(-2,-1))).flatten()
    # over_idx = np.argwhere((base_u8 >= 255).sum(axis=(-2,-1))).flatten()
    # # test
    # u = under_idx[0]
    # prev, cur = dct_blocks_backup[0, u], dct_blocks[0, u]
    # print((10*(dct_blocks_backup[0, u] - dct_blocks_backup[1, u])).astype(int))
    # print((prev*10).astype(int) - (cur*10).astype(int))
    return errors


def _calc_quantization_loss(base: np.ndarray, debug=False):
    assert base.dtype == float
    errors = {}
    # calc quantization loss
    residual = base - np.round(base)
    errors['single pixel'] = abs(residual).mean()
    if debug:
        print('Quantization loss(single pixel loss):', errors['Quantization loss(single pixel loss)'])
    return errors


def _to_blocks(src: np.ndarray, block_size=_default_block_size):
    assert len(src.shape) == 3
    # split img as blocks
    M, N = block_size, block_size
    shape = src.shape[0]//M, src.shape[1]//N, M, N
    blocks = src.reshape((shape[0], M, shape[1], N, src.shape[-1])).transpose(4, 0, 2, 1, 3)
    blocks = blocks.reshape(src.shape[-1], blocks.shape[1]*blocks.shape[2], M, N)
    return blocks   # 3xrowxcolx8x8


def _from_blocks(blocks: np.ndarray, genesis_shape):
    assert len(genesis_shape) == 3
    # construct img using img blocks
    M, N = blocks.shape[-2], blocks.shape[-1]
    shape = genesis_shape[2], genesis_shape[0]//M, genesis_shape[1]//N, M, N
    img = blocks.reshape(shape)    # reshape block to split block size
    img = img.transpose(0, 1, 3, 2, 4) # recover block
    img = img.transpose(1,2,3,4,0)
    img = img.reshape(genesis_shape)
    return img


def _to_pixel_range(x):
    # # scale to legal region
    # # if > 128, then scale to (min, 128)
    # # if < -127, then scale to (-127, max)
    # over = (x>128).sum(axis=(-2,-1)) > 0
    # low = x[over].min(axis=(-2,-1))[:, None, None]
    # x[over] = (x[over] - low) / x[over].max(axis=(-2, -1))[:, None, None] * (128-low) + low
    # under = (x<-127).sum(axis=(-2,-1)) > 0
    # top = x[under].max(axis=(-2,-1))[:, None, None]
    # x[under] = (x[under] - top) / x[under].min(axis=(-2, -1))[:, None, None] * (-127+top) + top

    # truncate extra part
    shape = x.shape
    x = x.flatten()
    cond = x > 128; x[cond] = 128
    cond = x <-127; x[cond] = -127
    cond = x > 0; x[cond] = np.floor(x[cond])
    cond = x < 0; x[cond] = np.ceil(x[cond])

    return x.reshape(shape)
# _to_pixel_range = np.vectorize(_to_pixel_range)


def _dwt2(ar: np.ndarray):
    a, (h,v,d) = pywt.dwt2(ar, 'haar')
    return np.array([a,h,v,d])


def _preprocess(src: np.ndarray, wm=None, jpeg_block_size=None):
    if isinstance(src, str):
        src = cv.imread(src)
        assert src is not None, 'read image failed: {}'.format(src)
    assert len(src.shape) >= 3, 'only support colorful image'

    jpeg_block_size = jpeg_block_size or _default_block_size
    wm_bits = None
    if wm is None:
        pass
    elif isinstance(wm, np.ndarray):
        wm_bits = wm
    elif isinstance(wm, bytes):
        wm_bits = np.unpackbits(np.array(list(wm), dtype=np.uint8))
        img_max_capacity = np.prod([x // jpeg_block_size for x in src.shape[:2]])
        assert len(wm_bits) <= img_max_capacity, \
            'wm_bits > img_max_capacity: {} > {}'.format(len(wm_bits), img_max_capacity)
    else:
        raise TypeError('wm serializer only support `bytes`')

    # get trimmed shape of src
    rows, cols = (np.array(src.shape[:2])//jpeg_block_size)*jpeg_block_size
    wm_size = wm_bits.size if wm_bits is not None else None
    return src, (rows, cols), wm_bits, wm_size, jpeg_block_size


_dctn = lambda ar: fftpack.dctn(ar, norm='ortho', axes=(-2,-1))
_idctn = lambda ar: fftpack.idctn(ar, norm='ortho', axes=(-2,-1))
_idwt2 = lambda ar: pywt.idwt2((ar[0], ar[1:]), 'haar')


# split to blocks, DWT DCT only
class DwtDctBlockWatermark(AbstractWatermark):
    _wavelet_component_index = 0  # a h v d

    def __init__(
            self,
            jpeg_block_size=_default_block_size,
            freq_band_to_be_embedded=6,
            wm_strength=400,
            debug=False,
    ):
        self.jpeg_block_size = jpeg_block_size
        self.freq_band_to_be_embedded = freq_band_to_be_embedded
        self.wm_strength = wm_strength
        self._debug = debug
        super().__init__()

    def embed(self, src: np.ndarray, wm, jpeg_block_size=None, measures=None, *args, **kwargs):
        # define
        f = self.freq_band_to_be_embedded // 2
        widx = self._wavelet_component_index  # wavelet horizontal
        src, (rows, cols), wm_bits, wm_size, block_size = _preprocess(src, wm, jpeg_block_size)
        if wm_bits.size == 0:
            return src

        # split img as blocks
        blocks = _to_blocks(src[:rows, :cols], block_size=block_size).astype(float) - 127
        block_indices = np.linspace(0, blocks.shape[1], wm_size, endpoint=False).astype(int)
        assert len(block_indices) == len(set(block_indices)), \
            'repeated img blocks index,shape={},wm_size={}'.format(src.shape, wm_size)
        block_indices_copy = block_indices.copy()
        wm_bits_copy = wm_bits.copy()

        # embed
        def _trans(ar: np.ndarray):
            blue, green = _dctn(_dwt2((ar))).transpose(1, 0, 2, 3, 4)
            return blue, green

        def _itrans(blue: np.ndarray, green: np.ndarray):
            res = _idctn(np.array([blue, green]).transpose(1, 0, 2, 3, 4))
            res = _idwt2(res)
            return res

        max_embedding_trying = 1
        blue = None
        for i in range(max_embedding_trying):
            if len(wm_bits) == 0:
                break
            # preprocess, force the image data to ordinary range
            v = np.argwhere((blocks[0, block_indices] > 120).sum(axis=(-2,-1)) > 0)
            blocks[0, block_indices[v]] -= 8
            v = np.argwhere((blocks[0, block_indices] < -120).sum(axis=(-2, -1)) > 0)
            blocks[0, block_indices[v]] += 7

            # DWT & DCT
            blue, green = _trans(blocks[:2, block_indices])

            # # 4 8 9
            # print(blocks[0, block_indices[2], :, :], wm_bits[2])
            # print(blue[widx, 2])

            # embed
            cB = (blue[widx, :, f, :]).sum(axis=-1) + (blue[widx, :, :, f]).sum(axis=-1)
            cG = (green[widx, :, f, :]).sum(axis=-1) + (green[widx, :, :, f]).sum(axis=-1)
            dB = cB - cG
            cond1 = np.bitwise_and(wm_bits == True, cB - self.wm_strength/2 <= cG)
            dB[cond1] = -dB[cond1] + self.wm_strength
            cond_1 = np.bitwise_and(wm_bits == False, cB + self.wm_strength/2 >= cG)
            dB[cond_1] = -dB[cond_1] - self.wm_strength
            # get original energy of freq band
            cond = np.bitwise_or(cond1, cond_1)
            energy = (blue[widx, cond, f, :]**2).sum(axis=-1) + (blue[widx, cond, :, f]**2).sum(axis=-1)
            energy_dc = (blue[widx, cond, 0, 0]**2)
            # get wm strength weight for each freq
            ## for dB > 0, get negative coefficient of freq bands, counteract with dB one be one, until dB == 0
            ## if dB still > 0 after that operation, add dB to the positive coefficients evenly
            ## if their got no positive coefficient, then calc residual amplitude with DC
            for i in range(dB.size):
                db = dB[i]
                cbv, cbh = blue[widx, i, f, :], blue[widx, i, :, f]
                if db > 0:
                    for n in range(cbv.size):
                        if cbv[n] < 0:
                            diff = min(abs(cbv[n]), db)
                            cbv[n] += diff
                            db -= diff
                        if cbh[n] < 0:
                            diff = min(abs(cbh[n]), db)
                            cbh[n] += diff
                            db -= diff
                        if db == 0:
                            break
                else:
                    for n in range(cbv.size):
                        if cbv[n] > 0:
                            diff = min(cbv[n], abs(db))
                            cbv[n] -= diff
                            db += diff
                        if cbh[n] > 0:
                            diff = min(cbh[n], abs(db))
                            cbh[n] -= diff
                            db += diff
                        if db == 0:
                            break
                # update
                # print('counteracted:', dB[i] - db)
                dB[i] = db
                blue[widx, i, f, :], blue[widx, i, :, f] = cbv, cbh

            weight = gaussian(f, sigma_x=1)(np.arange(blue.shape[-1]))
            weight /= (weight.sum() * 2)
            val = dB[:, None] @ weight[None, :]
            blue[widx, :, f, :] += val
            blue[widx, :, :, f] += val
            # balance energy
            energy_modified = (blue[widx, cond, f, :]**2).sum(axis=-1) + (blue[widx, cond, :, f]**2).sum(axis=-1)
            energy_diff = energy_modified - energy
            dc = energy_dc - energy_diff
            dc[dc < 0] = 0  # trim to
            dc = np.sqrt(dc)
            dc[blue[widx, cond, 0, 0] < 0] *= -1
            blue[widx, cond, 0, 0] = dc  # 359,285->300,261, proof in src.jpg & flower.jpg, benefit for precision loss

            # IDCT & IDWT
            blocks[:2, block_indices] = _itrans(blue, green)

            # Counteract deviation, rounding、uint8 overflow
            b = blocks[0, block_indices]
            cond = np.bitwise_or(cond1, cond_1)
            over_idx = np.argwhere(np.bitwise_and((b > 128).sum(axis=(-2, -1)) > 0, cond)).flatten()
            under_idx = np.argwhere(np.bitwise_and((b < -127).sum(axis=(-2, -1)) > 0, cond)).flatten()
            # print('overflow, underflow', over_idx.shape, under_idx.shape, under_idx[:3])
            precision_idx = np.hstack((over_idx, under_idx))
            blocks[0, block_indices[precision_idx]] = _to_pixel_range(blocks[0, block_indices[precision_idx]])
            # update
            block_indices = block_indices[precision_idx]
            wm_bits = wm_bits[precision_idx]

        # # test wm restore
        # blue, green = _trans(blocks[:2, block_indices_copy])
        # # embed
        # cB = (blue[widx, :, f, :]).sum(axis=-1) + (blue[widx, :, :, f]).sum(axis=-1)
        # cG = (green[widx, :, f, :]).sum(axis=-1) + (green[widx, :, :, f]).sum(axis=-1)
        # res = cB > cG
        # print('error rate:', (res != wm_bits_copy).sum(), len(wm_bits_copy))

        # compose blocks
        dst = src.copy()
        dst[:rows, :cols] = np.round(_from_blocks(blocks, (rows, cols, src.shape[2]))+127)
        if measures is not None:
            measures['QuantizationLoss'] = _calc_quantization_loss(blocks[0, block_indices_copy], debug=self._debug)
            measures['PrecisionLoss'] = _calc_uint8_precision_loss(blocks[0, block_indices_copy] + 127, debug=self._debug)
            measures['PSNR'] = [-1 if x == np.inf else x for x in ImageMeasure.PSNR(src, dst)]

        return dst

    def extract(self, src: np.ndarray, jpeg_block_size=None, wm_size=None, refer=None, measures=None, return_bin_array=False, *args, **kwargs):
        # split img as blocks
        f = self.freq_band_to_be_embedded//2
        widx = self._wavelet_component_index  # choose horizontal component
        src, (rows, cols), wm_refer, wm_size2, block_size = _preprocess(src, refer, jpeg_block_size)
        wm_size = wm_size or wm_size2
        blocks = _to_blocks(src[:rows, :cols], block_size=block_size).astype(float) - 127
        block_indices = np.linspace(0, blocks.shape[1], wm_size, endpoint=False).astype(int)

        # DWT & DCT
        blue, green = _dctn(_dwt2((blocks[:2, block_indices]))).transpose(1,0,2,3,4)
        # extract
        cB = (blue[widx, :, f, :]).sum(axis=-1) + (blue[widx, :, :, f]).sum(axis=-1)
        cG = (green[widx, :, f, :]).sum(axis=-1) + (green[widx, :, :, f]).sum(axis=-1)
        cond = cB > cG
        wm_bits = cond.astype(np.uint8)

        if wm_refer is not None:
            diff = wm_bits != wm_refer
            diff_idx = np.argwhere(diff)
            # print(diff_idx[:5], blue[widx, block_indices[2]], wm_refer[2])
            # print(np.argwhere(diff))
            ber = diff.sum() / diff.size
            if measures is not None:
                measures['BER'] = ber
            if self._debug:
                print('BitErrorRate:', ber)

        if return_bin_array:
            ret = wm_bits
        else:
            ret = np.packbits(wm_bits).tobytes()
        return ret

    def measure(self, src: np.ndarray, wm, jpeg_block_size=None, attack=lambda x: x, pretty=False, *args, **kwargs):
        # preprocess
        assert isinstance(wm, bytes), '`measure` only support wm whose type is bytes'
        if isinstance(src, str):
            src = cv.imread(src)

        # define
        measures = {}
        jpeg_block_size = jpeg_block_size or self.jpeg_block_size

        # embed
        dst = self.embed(src, wm, jpeg_block_size=jpeg_block_size, measures=measures)

        # simulate attack
        dst = attack(dst)
        assert src.shape == dst.shape, f'{self.__class__.__name__} require constant image size while attacking'

        # restore watermark
        wm_restore = self.extract(dst, jpeg_block_size=jpeg_block_size, wm_size=len(wm)*8, refer=wm, measures=measures)

        if pretty:
            print(json.dumps(measures, indent=2 if pretty else 0))

        #
        return measures


