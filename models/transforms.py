import time

import cv2 as cv
import numpy as np
import torch as tc
import pywt
from torch import nn
from scipy import fftpack
from torchvision.transforms import ToTensor, Compose
from PIL import Image
from .context import WatermarkContext


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


class ToSignalValueRange:
    def __init__(self, ctx: WatermarkContext):
        self.ctx = ctx

    def __call__(self, img):
        return self.forward(img)

    def forward(self, img):
        if isinstance(img, tc.Tensor):
            img = img.numpy()
        img[img > 1] = 1
        img[img < 0] = 0
        img -= 0.5
        return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class FromSignalValueRange:

    def __init__(self, ctx: WatermarkContext):
        self.ctx = ctx

    def __call__(self, img):
        return self.forward(img)

    def forward(self, img):
        if isinstance(img, tc.Tensor):
            img = img.numpy()
        img += 0.5
        img[img > 1] = 1
        img[img < 0] = 0
        return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class CropToBlocks:
    """Split Img to several JPEG blocks
    """

    def __init__(self, ctx: WatermarkContext):
        self.ctx = ctx

    def __call__(self, img):
        return self.forward(img)

    def forward(self, img):
        if isinstance(img, tc.Tensor):
            img = img.numpy()
        M, N = self.ctx.jpeg_block_size, self.ctx.jpeg_block_size
        rows, cols, row_remain, col_remain = img.shape[1]//M, img.shape[2]//N, img.shape[1]%M, img.shape[2]%N,
        assert len(img.shape) == 3
        self.ctx.image, self.ctx.image_block_shape = img, (rows, cols)

        # crop
        img = img[:, :rows*M, :cols*N]

        blocks = img.reshape((3, rows, M, cols, N))
        # blocks = blocks.transpose(0, 1, 3, 2, 4)
        blocks = blocks.swapaxes(2, 3)
        return blocks   # 3xrowxcolx8x8

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class FromBlocks:
    """Compose Img from several JPEG blocks
    """

    def __init__(self, ctx: WatermarkContext):
        self.ctx = ctx

    def __call__(self, img):
        return self.forward(img)

    def forward(self, blocks):
        M, N = self.ctx.jpeg_block_size, self.ctx.jpeg_block_size
        rows, cols = self.ctx.image_block_shape

        blocks = blocks.swapaxes(2, 3)
        img = blocks.reshape((3, rows*M, cols*N))
        self.ctx.dest_image = self.ctx.image.copy()
        self.ctx.dest_image[:, :rows*M, :cols*N] = img

        return self.ctx.dest_image

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class DiscreteWaveletTransform:
    """DWT for image
    """

    def __init__(self, ctx: WatermarkContext, axis=(-2, -1)):
        self.ctx = ctx
        self.axis = axis

    def __call__(self, blocks):
        return self.forward(blocks)

    def forward(self, blocks):
        a, (h, v, d) = pywt.dwt2(blocks, self.ctx.wavelet_name, mode='reflect', axes=self.axis)
        ret = np.array([a, h, v, d]).swapaxes(0, 1)
        return ret

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class InverseDiscreteWaveletTransform:
    """IDWT for image
    """

    def __init__(self, ctx: WatermarkContext, axis=(-2, -1)):
        self.ctx = ctx
        self.axis = axis

    def __call__(self, blocks):
        return self.forward(blocks)

    def forward(self, blocks):
        blocks = blocks.swapaxes(0, 1)
        a, (h, v, d) = blocks[0], blocks[1:]
        ret = pywt.idwt2((a, (h, v, d)), self.ctx.wavelet_name, mode='reflect', axes=self.axis)
        ret = ret
        return ret

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class DiscreteCosineTransform:
    """DFT for image
    """

    def __init__(self, ctx: WatermarkContext, axis=(-2, -1)):
        self.ctx = ctx
        self.axis = axis

    def __call__(self, blocks):
        return self.forward(blocks)

    def forward(self, blocks):
        freq = fftpack.dctn(blocks, norm='ortho', axes=self.axis)
        return freq

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class InverseDiscreteCosineTransform:
    """IDCT for image
    """

    def __init__(self, ctx: WatermarkContext, axis=(-2, -1)):
        self.ctx = ctx
        self.axis = axis

    def __call__(self, data):
        return self.forward(data)

    def forward(self, freq):
        spacial = fftpack.idctn(freq, norm='ortho', axes=self.axis)
        ret = spacial
        return ret

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class AbstractChannelReference:
    def __init__(self, ctx: WatermarkContext, channel_embed=2, channel_refer=1):
        super().__init__()
        self.ctx = ctx
        self.channel_embed, self.channel_refer = channel_embed, channel_refer  # RGB

    def select_blocks_to_be_embedded(self, blocks, wm_size, eval_choosing_error=False):
        # select indices evenly
        img_max_capacity = np.prod(self.ctx.image_block_shape)
        block_indices = np.linspace(0, img_max_capacity, wm_size, endpoint=False).astype(int)   # select blocks evenly
        block_indices += img_max_capacity - block_indices[-1] - 1

        # check & return
        if eval_choosing_error and self.ctx.block_indices is not None:
            err_rate = (block_indices != self.ctx.block_indices).sum()
            if err_rate != 0:
                print('choosing error:',
                      (block_indices != self.ctx.block_indices).sum(), block_indices[:10], self.ctx.block_indices[:10])
                self.ctx.errors['ChoosingErr'] = err_rate / block_indices.size
        else:
            self.ctx.block_indices = block_indices
        assert len(block_indices) == len(set(block_indices)), \
            'repeated img blocks index,shape={},wm_size={}'.format(blocks.shape, wm_size)
        return block_indices


class EmbedBitUsingChannelReference(AbstractChannelReference, tc.nn.Module):
    def __init__(self, ctx: WatermarkContext, print_error_block=None, use_numpy=True, **kwargs):
        super().__init__(ctx, **kwargs)
        self.print_error_block = print_error_block
        self.use_numpy = use_numpy

    def __call__(self, blocks):
        start_time = time.time()
        ret = self.forward(blocks)
        self.ctx.errors['EmbedElapsed'] = time.time() - start_time
        return ret

    def forward(self, blocks):
        img_max_capacity = np.prod(self.ctx.image_block_shape)
        if self.ctx.wm_bits is None:
            self.ctx.wm_bits = np.random.bytes(img_max_capacity // (9 * 8))  # insert 1 bit per 9 block
        if isinstance(self.ctx.wm_bits, bool):
            self.ctx.wm_bits = np.array([1 if self.ctx.wm_bits else 0] * img_max_capacity, dtype=np.uint8)
        wm_bits, wm_size = self.ctx.wm_bits, self.ctx.wm_size
        block_indices = self.select_blocks_to_be_embedded(blocks, wm_size)

        f = int(self.ctx.freq_band_to_be_embedded//2)
        weight = gaussian(f, sigma_x=42)(np.arange(blocks.shape[-1]))   # the end answer of the world
        weight /= (weight.sum() * 2 - weight[f])

        if self.print_error_block is not None:
            idx = self.print_error_block
            r, c = block_indices[idx] // blocks.shape[3], block_indices[idx] % blocks.shape[3]
            b = blocks[[self.channel_embed, self.channel_refer], self.ctx.wavelet_component_index, r, c]
            print(block_indices[idx], wm_bits[idx], '\t DCT block:\n', ((b[0]-b[1])*10000).astype(int))

        if self.use_numpy:
            # using numpy for more efficient performance
            tmp = blocks.reshape(blocks.shape[0], blocks.shape[1], -1, blocks.shape[-2], blocks.shape[-1])
            cB = tmp[self.channel_embed, self.ctx.wavelet_component_index, block_indices]
            cG = tmp[self.channel_refer, self.ctx.wavelet_component_index, block_indices]
            # embed: get reference
            cB = (cB[:, f, :]).sum(axis=-1) + (cB[:, :, f]).sum(axis=-1) - cB[:, f, f]
            cG = (cG[:, f, :]).sum(axis=-1) + (cG[:, :, f]).sum(axis=-1) - cG[:, f, f]
            dB = cB - cG
            cond1 = np.bitwise_and(wm_bits == True, cB - self.ctx.wm_strength / 2 <= cG)
            dB[cond1] = -dB[cond1] + self.ctx.wm_strength
            cond_1 = np.bitwise_and(wm_bits == False, cB + self.ctx.wm_strength / 2 >= cG)
            dB[cond_1] = -dB[cond_1] - self.ctx.wm_strength
            # embed: assign
            val = dB[:, None] @ weight[None, :]
            tmp[self.channel_embed, self.ctx.wavelet_component_index, block_indices, f, :] += val
            tmp[self.channel_embed, self.ctx.wavelet_component_index, block_indices, :, f] += val
            blocks = tmp.reshape(blocks.shape)
        else:
            # using for loop for more human readability
            for i, bi in enumerate(block_indices):
                r, c = bi // blocks.shape[3], bi % blocks.shape[3]
                embed, refer = blocks[[self.channel_embed, self.channel_refer], self.ctx.wavelet_component_index, r, c]
                # get cB freq band energy
                eB = (embed[f, :]**2).sum() + (embed[:, f]**2).sum() - embed[f, f]**2
                eDC = embed[0, 0]**2
                # get coeff val
                cB = (embed[f, :] + embed[:, f]).sum() - embed[f, f]
                cG = (refer[f, :] + refer[:, f]).sum() - refer[f, f]
                dB = cB - cG
                bit = wm_bits[i]
                val = 0
                if bit == True and cB - self.ctx.wm_strength/2 <= cG:
                    val = -dB + self.ctx.wm_strength
                elif bit == False and cB + self.ctx.wm_strength/2 >= cG:
                    val = -dB - self.ctx.wm_strength
                # embed
                val = val * weight
                embed[f, :] += val
                embed[:, f] += val
                embed[f, f] -= val[f]
                # calc energy residual
                eB_after = (embed[f, :]**2).sum() + (embed[:, f]**2).sum() - embed[f, f]**2
                dc = eDC - (eB_after - eB)
                dc = np.sqrt(dc if dc > 0 else 0)
                embed[0, 0] = dc if embed[0, 0] > 0 else -dc
                # update
                blocks[self.channel_embed, self.ctx.wavelet_component_index, r, c] = embed

        # # extract immediately
        # f = int(self.ctx.freq_band_to_be_embedded//2)
        # wm_restore = []
        # for i, bi in enumerate(block_indices):
        #     r, c = bi // blocks.shape[3], bi % blocks.shape[3]
        #     embed, refer = blocks[[self.channel_embed, self.channel_refer], self.ctx.wavelet_component_index, r, c]
        #     # get coeff val
        #     cB = (embed[f, :] + embed[:, f] - embed[f, f]).sum()
        #     cG = (refer[f, :] + refer[:, f] - refer[f, f]).sum()
        #     # save bit
        #     bit_restore = (cB - cG)
        #     wm_restore.append(bit_restore)
        # wm_restore = np.array(wm_restore) > 0
        # print('error rate:', (wm_restore.astype(np.uint8) != wm_bits).sum(), wm_bits.size)

        return blocks

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class ExtractBitUsingChannelReference(AbstractChannelReference, tc.nn.Module):
    def __init__(self, ctx: WatermarkContext, print_error_block=None, use_numpy=True, **kwargs):
        super().__init__(ctx, **kwargs)
        self.print_error_block = print_error_block
        self.use_numpy = use_numpy

    def __call__(self, blocks):
        return self.forward(blocks)

    def forward(self, blocks):
        wm_bits, wm_size = self.ctx.wm_bits, self.ctx.wm_size
        block_indices = self.select_blocks_to_be_embedded(blocks, wm_size, eval_choosing_error=True)

        f = int(self.ctx.freq_band_to_be_embedded//2)
        if self.use_numpy:
            # using numpy for more efficient performance
            tmp = blocks.reshape(blocks.shape[0], blocks.shape[1], -1, blocks.shape[-2], blocks.shape[-1])
            cB = tmp[self.channel_embed, self.ctx.wavelet_component_index, block_indices]
            cG = tmp[self.channel_refer, self.ctx.wavelet_component_index, block_indices]
            # embed: get reference
            cB = (cB[:, f, :]).sum(axis=-1) + (cB[:, :, f]).sum(axis=-1) - cB[:, f, f]
            cG = (cG[:, f, :]).sum(axis=-1) + (cG[:, :, f]).sum(axis=-1) - cG[:, f, f]
            dB = cB - cG
            wm_restore = dB
        else:
            wm_restore = []
            for i, bi in enumerate(block_indices):
                r, c = bi // blocks.shape[3], bi % blocks.shape[3]
                embed, refer = blocks[[self.channel_embed, self.channel_refer], self.ctx.wavelet_component_index, r, c]
                # get coeff val
                cB = (embed[f, :] + embed[:, f]).sum() - embed[f, f]
                cG = (refer[f, :] + refer[:, f]).sum() - refer[f, f]
                # save bit
                bit_restore = (cB - cG)
                wm_restore.append(bit_restore)

                if self.print_error_block is not None and i == self.print_error_block:
                    idx = self.print_error_block
                    r, c = block_indices[idx] // blocks.shape[3], block_indices[idx] % blocks.shape[3]
                    b = blocks[[self.channel_embed, self.channel_refer], self.ctx.wavelet_component_index, r, c]
                    diff = (b[0]-b[1])
                    print(block_indices[idx], int(cB*10000), int(cG*10000),
                          '\t attacked DCT block:\n', (diff*10000).astype(int))
            wm_restore = np.array(wm_restore)
        if wm_bits is not None:
            ber = ((wm_bits > 0) != (wm_restore > 0))
            ber_idx = np.argwhere(ber).flatten()
            self.ctx.block_indices_caused_error_bit = ber_idx
            ber = ber.sum() / ber.size
            self.ctx.errors['BER'] = ber

        return wm_restore

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class CollectStatistic:
    def __init__(self, ctx: WatermarkContext):
        self.ctx = ctx

    def __call__(self, img):
        return self.forward(img)

    def forward(self, img):
        assert img.shape == self.ctx.image.shape, \
            'img not match original img shape: {} != {}'.format(img.shape, self.ctx.image.shape)

        # get PSNR
        img_size, wm_size = np.prod(img.shape[-2:]), self.ctx.wm_size
        a, b = (img+0.5) * 255, (self.ctx.image+0.5) * 255
        A, B = np.round(a), np.round(b)
        A[A > 255] = 255
        A[A < 0] = 0
        loss = nn.MSELoss()
        mse = tc.tensor([loss(tc.tensor(A[i]), tc.tensor(B[i])) for i in range(3)])
        with np.errstate(divide='ignore'):
            psnr = 10 * tc.log10(255**2/mse)
        self.ctx.errors['PSNR'] = psnr

        # get overflow, underflow rate
        overflow_rate = [(x > 255).sum() / wm_size for x in a]
        underflow_rate = [(x < 0).sum() / wm_size for x in a]
        self.ctx.errors['overflow'] = overflow_rate
        self.ctx.errors['underflow'] = underflow_rate

        return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class JPEGCompression:
    def __init__(self, ctx: WatermarkContext, jpeg_quality=80, to_file=None):
        self.ctx = ctx
        self.jpeg_quality = jpeg_quality
        self.to_file = to_file

    def __call__(self, img):
        return self.forward(img)

    def forward(self, img):
        img[img > 1] = 1
        img[img < 0] = 0
        img = (img * 255).astype(np.uint8).transpose(1, 2, 0)[:, :, [2, 1, 0]]
        if self.to_file:
            ok = cv.imwrite(self.to_file, img, [cv.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            assert ok, 'cv.imwrite to file failed: {}'.format(self.to_file)
            img = cv.imread(self.to_file)
            assert img is not None, 'cv.imread failed: {}'.format(self.to_file)
        else:
            ok, data = cv.imencode('.jpg', img, [cv.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            assert ok == True, 'encode img to jpg failed'
            img = cv.imdecode(data, cv.IMREAD_UNCHANGED)
        img = img.transpose(2, 0, 1)[[2, 1, 0]].astype(float) / 255

        return tc.tensor(img)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


def select_image_area(img: np.ndarray, savefile='', max_diameter=False):
    orb = cv.ORB_create(scaleFactor=1.1, nlevels=16)
    # find the key points which have the biggest response in their region respectively
    kp = list(orb.detect(img, None))
    for i in range(len(kp)-1):
        if kp[i] is None:
            continue
        for n in range(len(kp)-1-i):
            if kp[i+1+n] is None:
                continue
            if cv.KeyPoint.overlap(kp[i], kp[i+1+n]) != 0:
                if kp[i].response >= kp[i+1+n].response:
                    kp[i+1+n] = None
                else:
                    kp[i] = None
                    break
    kp = sorted(filter(lambda x: x is not None, kp), key=lambda x: x.response, reverse=True)

    # plot and save
    if savefile:
        img2 = cv.drawKeypoints(img, kp[:], None, color=(0,255,0), flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv.imwrite(savefile, img2)

    # for i in range(4):
    #     k = sorted_kp[i]
    #     print(k.size, k.pt, k.octave, k.response)

    # key point to area list
    coeff = np.sqrt(2)
    pt = np.array([[x.pt[1], x.pt[0], x.size if max_diameter else x.size/coeff] for x in kp])
    pt = np.round([(x[0]-x[2]/2, x[1]-x[2]/2, x[2]) for x in pt]).astype(int)
    return pt


def select_image_area2(img: np.ndarray, savefile='', diameter=8*np.sqrt(2)):
    def overlap(x, y):
        x = np.array(x.pt)
        y = np.array(y.pt)
        return int(np.sqrt(((x - y)**2).sum()) <= diameter)

    orb = cv.ORB_create(scaleFactor=1.1, nlevels=16)
    # find the key points which have the biggest response in their region respectively
    kp = list(orb.detect(img, None))
    for i in range(len(kp)-1):
        if kp[i] is None:
            continue
        for n in range(len(kp)-1-i):
            if kp[i+1+n] is None:
                continue
            if overlap(kp[i], kp[i+1+n]) != 0:
                if kp[i].response >= kp[i+1+n].response:
                    kp[i+1+n] = None
                else:
                    kp[i] = None
                    break
    kp = sorted(filter(lambda x: x is not None, kp), key=lambda x: x.response, reverse=True)

    # plot and save
    if savefile:
        img2 = cv.drawKeypoints(img, kp[:], None, color=(0,255,0))
        cv.imwrite(savefile, img2)

    # for i in range(4):
    #     k = sorted_kp[i]
    #     print(k.size, k.pt, k.octave, k.response)

    # # key point to area list
    # pt = np.array([[x.pt[1], x.pt[0]] for x in kp])
    # area = np.zeros((pt.shape[0], 8, 8, 3))
    # for i, (r, c) in enumerate(np.round(pt[:, :2] - 4).astype(int)):
    #     area[i] = (img[r:r+8, c:c+8])
    # a, _ = pywt.dwt2(area, 'haar', axes=(1, 2))
    # freq = fftpack.dctn(a, norm='ortho', axes=(1, 2))
    # # print(np.percentile(abs(freq), (0, 25, 50, 75, 100)), freq.mean())
    # Ef = np.abs(freq).sum(axis=(1, 2, 3)) / 48   # sum of all coefficients
    # Rf = (np.abs(freq) > 20).sum(axis=(1, 2, 3)) / 48    # possibility each coefficient is not 0
    # # print(Rf.shape, Rf.mean(), np.percentile(Rf, (0, 25, 50, 75, 100)))
    # Si = Ef + Rf + Ef*Rf
    # # sorted_idx = sorted(range(len(kp)), key=lambda x: Si[x], reverse=True)
    # # print(Si.shape, np.percentile(Si, (0, 25, 50, 75, 100)))
    #
    # # 由小到大、由左上至右下、由 response 从大到小排序
    # def kp_sort_key(i):
    #     nonlocal kp
    #     ki = kp[i]
    #     si = Si[i]
    #     ret = ki.pt[0] * img.shape[1] + ki.pt[1]
    #     # print(ret, ki.size, ki.response*100, ki.pt[0])
    #     ret *= (ki.size + ki.response*100)
    #     return ret
    # kp_sorted_index = sorted(np.arange(len(kp)), key=kp_sort_key, reverse=True)
    pt = np.round([[x.pt[1], x.pt[0], 8.] for x in kp]).astype(int)

    return pt


def embed_one_block(b: np.ndarray, bit: bool, f=6, strength=400):
    b = b.astype(float)
    b -= 127
    a, coeff = pywt.dwt2(b.astype(float), 'haar', axes=(0, 1))
    q = fftpack.dctn(a, norm='ortho', axes=(0,1))

    # embed
    f //= 2
    cG, cB = ((q[f, :, 1:] + q[:, f, 1:]).sum() - q[f, f, 1:][None, None,:]).transpose(2, 0, 1)
    cG, cB = cG.flatten()[0], cB.flatten()[0]
    dB = cB - cG
    if bit == True and cB - strength/2 <= cG:
        dB = -dB + strength
    elif bit == False and cB + strength/2 >= cG:
        dB = -dB - strength
    weight = np.ones(4)
    weight /= weight.size*2-1
    weight[f] = weight[f]/2
    val = dB * weight
    # q[f, :, 2] += val
    # q[:, f, 2] += val

    a1 = fftpack.idctn(q, norm='ortho', axes=(0,1))
    b1 = pywt.idwt2((a1, coeff), 'haar', axes=(0,1))
    b1 = np.round(b1)
    b1[b1 > 128] = 128
    b1[b1 < -127] = -127
    b1 += 127
    b1 = b1.astype(np.uint8)
    return b1


class AdaptiveRegionSelectionWatermark(nn.Module):
    INPUT_FORMAT_TENSOR = 0  # 3*Height*Width
    INPUT_FORMAT_PIL = 1    # PIL.Image object
    INPUT_FORMAT_RGB = 2    # Height*Width*3
    INPUT_FORMAT_BGR = 3    # Height*Width*3

    OUTPUT_FORMAT_RGB = 0
    OUTPUT_FORMAT_TENSOR = 1

    @classmethod
    def get_embeder(cls, ctx: WatermarkContext, print_error_block=None):
        return Compose([
            ToTensor(),
            ToSignalValueRange(ctx),
            CropToBlocks(ctx),
            DiscreteWaveletTransform(ctx),
            DiscreteCosineTransform(ctx),
            EmbedBitUsingChannelReference(ctx, print_error_block=print_error_block),
            InverseDiscreteCosineTransform(ctx),
            InverseDiscreteWaveletTransform(ctx),
            FromBlocks(ctx),
            FromSignalValueRange(ctx),
            CollectStatistic(ctx),
        ])

    @classmethod
    def get_extractor(cls, ctx: WatermarkContext, print_error_block=None):
        return Compose([
            ToTensor(),
            ToSignalValueRange(ctx),
            CropToBlocks(ctx),
            DiscreteWaveletTransform(ctx),
            DiscreteCosineTransform(ctx),
            ExtractBitUsingChannelReference(ctx, print_error_block=print_error_block),
        ])

    def __init__(self, ctx: WatermarkContext, is_embed=True, save_kp_img_to='', best_performance=True,
                 input_format=INPUT_FORMAT_RGB, output_format=OUTPUT_FORMAT_RGB, inplace=True):
        self.ctx = ctx
        self.save_kp_img_to = save_kp_img_to
        self.is_embed = is_embed
        self.best_performance = best_performance
        self.input_format, self.output_format, self.inplace = input_format, output_format, inplace
        super().__init__()

    def __call__(self, img):
        # unify input format
        if self.input_format == self.INPUT_FORMAT_PIL:
            img = np.array(img)
        elif self.input_format == self.INPUT_FORMAT_TENSOR:
            img = np.transpose(img.numpy() * 255, (1, 2, 0)).astype(np.uint8)

        start_time = time.time()
        ret = self.forward(img)
        self.ctx.errors['TotalElapsed'] = time.time() - start_time
        return ret

    def forward(self, img):
        if self.is_embed:
            dest = self.embed(img)
            self.ctx.image, self.ctx.dest_image = img, dest
            if not self.best_performance:
                # PSNR
                psnr = ImageMeasure.PSNR(img, dest)
                print(psnr, 'psnr')
                self.ctx.errors['PSNR'] = psnr

            # unify output format
            if self.output_format == self.OUTPUT_FORMAT_RGB:
                ret = dest
            elif self.output_format == self.OUTPUT_FORMAT_TENSOR:
                ret = np.transpose(dest/255, (2, 0, 1))
            else:
                raise AttributeError('unrecognized output format:{}'.format(self.output_format))
            return ret
        else:
            wm_restore = self.extract(img)
            if isinstance(self.ctx.wm_bits, np.ndarray):
                # BER
                assert len(wm_restore) == self.ctx.wm_size, 'wm_size not match after extraction'
                ber = ((wm_restore > 0) == (self.ctx.wm_bits)).sum() / self.ctx.wm_size
                self.ctx.errors['BER'] = ber
            elif isinstance(self.ctx.wm_bits, bool):
                self.ctx.errors['Confidence'] = wm_restore
            return wm_restore

    def embed(self, img):
        start_time = time.time()
        wm_bits, wm_size = self.ctx.wm_bits, self.ctx.wm_size
        assert wm_bits is not None, 'wm_bits should not be None when embedding'
        pt = select_image_area(img, savefile=self.save_kp_img_to)[:9]
        self.ctx.errors['ORBElapsed'] = time.time() - start_time
        context, bits_count, dest = [], 0, (img if self.inplace else img.copy())
        for r, c, width in pt:
            a, capacity = img[r:r+width, c:c+width], (width//8)**2
            if isinstance(wm_bits, bool):
                # from utils import scale2, iscale2
                # b, diff = scale2(a, (8,8,3))
                # b1 = embed_one_block(b, wm_bits, f=self.ctx.freq_band_to_be_embedded)
                # # print(np.abs(b-b1).sum())
                # a1 = iscale2(b1, diff)
                # a1 = np.round(a1)
                # a1[a1 > 255] = 255
                # a1[a1 < 0] = 0
                # dest[r:r+width, c:c+width] = a1
                # continue
                wm = wm_bits
            else:
                wm = wm_bits[bits_count:bits_count+capacity]
            ctx = WatermarkContext(wm_strength=self.ctx.wm_strength, wm_bits=wm)
            dest[r:r+width, c:c+width] = self.get_embeder(ctx)(a).transpose(1, 2, 0) * 255
            # update
            context.append(ctx)
            bits_count += capacity
            if not isinstance(wm_bits, bool) and bits_count >= wm_bits.size:
                break

        self.ctx.sub_context = context
        return dest

    def extract(self, img):
        wm_bits, wm_size = self.ctx.wm_bits, self.ctx.wm_size
        if not isinstance(self.ctx.wm_bits, bool):
            assert wm_size is not None, 'wm_size should not be None when extracting'
        pt = select_image_area(img[:, :, [2, 1, 0]], savefile=self.save_kp_img_to)[:9]
        if len(pt) == 0:
            cv.imwrite('data/dst.jpg', img[:, :, [2, 1, 0]])
        assert len(pt) > 0, 'get Key point from frame failed: {},{}'.format(img.shape, img.dtype)
        context, wm_bits_restore, capacity = [], [], 0
        for p in pt:
            r, c, width = np.round(p).astype(int)
            a, cp = img[r:r+width, c:c+width], (width//8)**2
            ctx = WatermarkContext(wm_size=cp)
            bits = self.get_extractor(ctx)(a)
            # update
            wm_bits_restore.extend(bits)
            context.append(ctx)
            capacity += cp
        self.ctx.sub_context = context
        self.wm_bits_restore = np.array(wm_bits_restore[:wm_size or capacity])
        # 这里应该画个直方图看下出错帧的蓝色通道分量分布，然后对应调整算法
        # bit = (self.wm_bits_restore > 0).sum() / len(self.wm_bits_restore)
        # return 1 if bit > 0.5 else 0
        return self.wm_bits_restore.mean()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class ToFormat:
    INPUT_FORMAT_TENSOR = 0  # 3*Height*Width
    INPUT_FORMAT_PIL = 1    # PIL.Image object
    INPUT_FORMAT_RGB = 2    # Height*Width*3
    INPUT_FORMAT_BGR = 3    # Height*Width*3

    OUTPUT_FORMAT_RGB = 0
    OUTPUT_FORMAT_BGR = 2
    OUTPUT_FORMAT_TENSOR = 1

    def __init__(self, input_format=INPUT_FORMAT_PIL, output_format=OUTPUT_FORMAT_BGR):
        self.input_format = input_format
        self.output_format = output_format

    def __call__(self, img):
        return self.forward(img)

    def forward(self, img):
        # unify format to numpy RGB24
        if self.input_format == self.INPUT_FORMAT_PIL:
            img = np.array(img)
        else:
            raise AttributeError('please add related input format support')

        # output
        if self.output_format == self.OUTPUT_FORMAT_BGR:
            img = img[:, :, [2, 1, 0]]
        else:
            raise AttributeError('please add related output format support')

        return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

