import numpy as np
import cv2 as cv


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


class AbstractWatermark:
    def embed(self, src: np.ndarray, wm, *args, **kwargs):
        pass

    def extract(self, src: np.ndarray, *args, **kwargs):
        pass


class BlindWatermark(AbstractWatermark):
    def __init__(self, *args, **kwargs):
        import blind_watermark
        blind_watermark.bw_notes.close()
        self.bwm = blind_watermark.WaterMark(*args, **kwargs)
        super().__init__()

    def embed(self, src: np.ndarray, wm, *args, wm_content=None, measures=None, **kwargs):
        # 读取原图
        self.bwm.read_img(img=src)
        # 读取水印
        self.bwm.read_wm(wm_content)
        # 打上盲水印
        dst = self.bwm.embed()
        self.wm_shape = wm.shape

        # measure wm effect
        if measures is not None:
            psnr = ImageMeasure.PSNR(src, dst)
            measures['PSNR'] = psnr

        return dst

    def extract(self, src: np.ndarray, *args, wm_shape=None, **kwargs):
        # 注意需要设定水印的长宽wm_shape
        wm_restore = self.bwm.extract(embed_img=src, wm_shape=wm_shape or self.wm_shape, out_wm_name='wm_restore.png')
        return wm_restore

    def measure(self, src: np.ndarray, wm, *args, wm_content=None, attack=lambda x: x):
        measures = {}

        # embed
        dst = self.embed(src, wm, wm_content=wm_content, measures=measures)

        # simulate attack
        dst = attack(dst)
        assert src.shape == dst.shape, 'blind_watermark require constant image size while attacking'

        # restore watermark
        wm_restore = self.extract(dst, out_wm_name='wm_restore.png')

        # measure
        ber = ImageMeasure.BER(wm > 127, wm_restore > 127)
        measures['BER'] = ber

        #
        return measures


def jpeg_compression_attack(where_to_save, jpeg_quality=70):
    def attack(img: np.ndarray):
        cv.imwrite(where_to_save, img, [cv.IMWRITE_JPEG_QUALITY, jpeg_quality])
        dst = cv.imread(where_to_save)
        return dst
    return attack


def random_scale_attack(scale_range=(1, 3), interpolation='linear'):
    def attack(img: np.ndarray):
        nonlocal scale_range
        scale = scale_range[0] + np.random.random(1)[0] * (scale_range[1] - scale_range[0])
        r, c = np.round(np.array(img.shape[:2]) / scale).astype(int)
        dst = cv.resize(img, (c, r), cv.INTER_LINEAR)
        dst = cv.resize(dst, img.shape[:2][::-1], cv.INTER_LINEAR)
        return dst
    return attack


def scale_attack(scale=2, interpolation='linear'):
    def attack(img: np.ndarray):
        nonlocal scale
        r, c = np.round(np.array(img.shape[:2]) / scale).astype(int)
        dst = cv.resize(img, (c, r), cv.INTER_LINEAR)
        dst = cv.resize(dst, img.shape[:2][::-1], cv.INTER_LINEAR)
        return dst
    return attack


def chain(*attacks):
    def attack(img: np.ndarray):
        for at in attacks:
            img = at(img)
        return img
    return attack
