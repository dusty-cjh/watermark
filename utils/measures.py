import numpy as np


class ImageMeasure:
    @classmethod
    def MSE(cls, src: np.ndarray, dst: np.ndarray, channel=None):
        if channel:
            src, dst = src[:, :, channel], dst[:, :, channel]
        mse = ((src - dst)**2).sum(axis=(0,1)) / np.prod(src.shape[:2])
        return mse

    @classmethod
    def PSNR(cls, src: np.ndarray, dst: np.ndarray, channel=None):
        if channel:
            src, dst = src[:, :, channel], dst[:, :, channel]
        with np.errstate(divide='ignore'):
            psnr = 10*np.log10(255**2 / cls.MSE(src, dst))
        return psnr

    @classmethod
    def BER(cls, src: np.ndarray, dst: np.ndarray, channel=None):
        assert src.dtype == bool and dst.dtype == bool, 'dtype of src and dst must be `bool`'
        if channel:
            src, dst = src[:, :, channel], dst[:, :, channel]
        return abs(src != dst).sum(axis=(0,1)) / np.prod(src.shape[:2])

    @classmethod
    def get_measures(cls, src: np.ndarray, dst: np.ndarray, channel=None, psnr=True, ber=True):
        ret = []
        if channel:
            src, dst = src[:, :, channel], dst[:, :, channel]
        if psnr:
            ret.append(cls.PSNR(src, dst))
        if ber:
            ret.append(cls.BER(src, dst))
        return ret


