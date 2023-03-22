import math
import numpy as np
import cv2 as cv
import ujson as json
import pywt
from scipy import fftpack, ndimage, interpolate, cluster
from .dwt_dct import AbstractWatermark, ImageMeasure, DwtDctBlockWatermark, _default_block_size, _preprocess


def _split_img_blocks(img: np.ndarray, ax=None, savefile='', max_diameter=False):
    # Initiate ORB detector
    scaleFactor = 1.1
    max_level = int(math.log(min(img.shape[:2]), scaleFactor) - math.log(31, scaleFactor))
    orb = cv.ORB_create(nfeatures=int(1e4), nlevels=int(max_level), scaleFactor=scaleFactor, firstLevel=0)

    # warning! NMS is a bad behavior, will deprecate soon.
    # find the key points which have the biggest response in their region respectively
    size_min, size_max = 9*_default_block_size*np.sqrt(2), min(img.shape[:2])/6
    kp = list(filter(lambda x: x.response > 0.0006 and size_min <= np.round(x.size) <= size_max, orb.detect(img,None)))
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
    kp = list(filter(lambda x: x is not None, kp))
    # print('kp max response:', sorted(kp, key=lambda x: x.response, reverse=True)[0].response)

    # get square area block indices and area itself
    pt = [np.array((x.pt[1], x.pt[0], x.size if max_diameter else x.size/np.sqrt(2))) for x in kp]
    pt = np.array([(x[0]-x[2]/2, x[1]-x[2]/2, x[2]) for x in pt])
    area = []
    for i, p in enumerate(np.round(pt).astype(int)):
        a = img[p[0]:p[0]+p[2], p[1]:p[1]+p[2]]
        area.append(a)

    # sort area by its robustness coefficient
    area_dct = [fftpack.dctn(a, axes=(0,1)) for a in area]
    Ef = np.array([np.abs(a).sum() / np.prod(a.shape[:2]) for a in area_dct])
    Rf = np.array([(np.abs(a) > 1).sum() / np.prod(a.shape[:2]) for a in area_dct])
    Si = Ef + Rf + Ef*Rf
    sorted_idx = sorted(range(len(kp)), key=lambda x: Si[x], reverse=True)

    # plot and save
    img2 = cv.drawKeypoints(img, kp[:], None, color=(0,255,0), flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    if ax is not None:
        ax.imshow(img2);ax.axis('off')
    if savefile:
        cv.imwrite(savefile, img2)

    sorted_pt, sorted_area, sorted_kp = list(zip(*[(pt[idx], area[idx], kp[idx])for idx in sorted_idx]))
    # for i in range(4):
    #     k = sorted_kp[i]
    #     print(k.size, k.pt, k.octave, k.response)
    return sorted_pt, sorted_area, sorted_kp


class AdaptiveRegionSelectionWatermark(DwtDctBlockWatermark):
    def __init__(self, rotation_attack=False, max_area_count=3, *args, **kwargs):
        assert rotation_attack == False, f'rotation feature temporary deprecated in {self.__class__.__name__}'
        self._rotation_attack = rotation_attack  # temporary deprecated
        self._max_area_count = max_area_count
        super().__init__(*args, **kwargs)

    def embed(self,
              src: np.ndarray, wm,
              jpeg_block_size=None, measures=None,
              ax=None, savefile='', inplace=False, return_context=False,
              *args, **kwargs):
        # preprocess input parameters
        src, trimmed_shape, wm_bits, wm_size, block_size = _preprocess(src, wm, jpeg_block_size)

        # get the top left coordinate and diameter of the split images
        pt, area, kp = _split_img_blocks(src, ax=ax, savefile=savefile, max_diameter=self._rotation_attack)
        context = {
            'pt': pt,
            'area': area,
        }

        # embed related wm block
        capacity, dst = 0, src if inplace else src.copy()
        m_list = {}
        for i in range(len(area)):
            p, a, k = pt[i], area[i], kp[i]
            r, c, size = np.round(p).astype(int)
            # calc capacity of this block
            inner_width = int(np.round(k.size / np.sqrt(2)))
            # cp = np.prod(np.array(a.shape[:2])//block_size)
            cp = (inner_width//8)**2
            if cp == 0:
                continue
            a_measure = dict(index_emb=i, capacity_emb=cp)

            # print('emb', inner_width, cp)

            # embed info
            emb = super().embed(
                a, wm_bits[capacity:capacity+cp],
                jpeg_block_size=jpeg_block_size, measures=a_measure, *args, **kwargs)
            dst[r:r+size, c:c+size] = emb

            # update var
            m_list.setdefault(f'{r},{c},{size}', {}).update(a_measure)
            capacity += cp
            if capacity >= wm_bits.size:
                break
        assert capacity >= wm_bits.size, 'reached img max capacity: {} < {}'.format(capacity, wm_bits.size)

        #
        if measures is not None:
            measures['sub'] = m_list
            # measures['region'] = pt

        ret = (dst, context) if return_context else dst
        return ret

    def extract(self, src: np.ndarray,
                jpeg_block_size=None, wm_size=None, refer=None,
                measures=None, ax=None, savefile='', context=None,
                *args, **kwargs):
        # preprocess input parameters
        src, trimmed_shape, wm_refer, wm_size, block_size = _preprocess(src, refer, jpeg_block_size)
        if wm_size is None:
            raise AttributeError('watermark size should not be None')

        # measures
        # 1. block selection -> sequence & set
        # 2. BER per block
        # 3. PSNR per block
        measures = measures if measures is not None else {}

        # get the top left coordinate and diameter of the split images
        pt, area, kp = _split_img_blocks(src, savefile=savefile, ax=ax, max_diameter=self._rotation_attack)

        # extract related wm block
        capacity, wm_bits, i = 0, [], 0
        m_list = measures.setdefault('sub', {})
        for i in range(len(area)):
            p, a, k = pt[i], area[i], kp[i]
            r, c, size = np.round(p).astype(int)
            # calc capacity of this block
            cp = np.prod(np.array(a.shape[:2])//block_size)
            if cp == 0:
                continue
            a_wm_size = min(cp, wm_size-capacity)
            a_measure = dict(index_ext=i)

            # print('ext', r,c,size,a.shape)

            # calc capacity of this block
            # extract
            bits = super().extract(
                a,
                jpeg_block_size=block_size,
                wm_size=a_wm_size,
                measures=a_measure,
                return_bin_array=True,
                refer=wm_refer[capacity:capacity+cp],
                *args, **kwargs)
            assert a_wm_size == len(bits), 'DwtDct extract block bits failure'
            wm_bits.extend(bits)
            # update var
            m_list.setdefault(f'{r},{c},{size}', {}).update(a_measure)
            capacity += cp
            if capacity >= wm_size:
                break
        wm_bits = np.array(wm_bits[:wm_size])

        # calc BER
        if refer:
            measures['BER'] = (wm_refer != wm_bits).sum() / wm_bits.size
        measures['sub'] = m_list
        count = min(len(pt), len(context['pt']))
        pt, cpt = pt[:i+1], context['pt'][:i+1]
        measures['ChoosingErrorRate'] = dict(
            sequence=((np.round(pt[:count]) != np.round(cpt[:count])).sum(axis=-1) > 0).sum()/len(pt),
            points=len(set([tuple(x) for x in np.round(pt)]) ^ set([tuple(x) for x in np.round(cpt)]))/len(pt),
        )

        return wm_bits

    def measure(self,
                src: np.ndarray, wm,
                jpeg_block_size=None, attack=lambda x: x, pretty=False, save_kp_img='',
                *args, **kwargs):
        # preprocess
        assert isinstance(wm, bytes), '`measure` only support wm whose type is bytes'
        if isinstance(src, str):
            src = cv.imread(src)
        measures = {}
        jpeg_block_size = jpeg_block_size or self.jpeg_block_size

        # embed
        dst, ctx = self.embed(
            src,
            wm,
            jpeg_block_size=jpeg_block_size,
            measures=measures,
            return_context=True,
            savefile=save_kp_img + '.emb.jpg' if save_kp_img else ''
        )
        measures['PSNR'] = [-1 if x == np.inf else x for x in ImageMeasure.PSNR(src, dst)]

        # simulate attack
        dst = attack(dst)
        assert src.shape == dst.shape, f'{self.__class__.__name__} require constant image size while attacking'

        # restore watermark
        wm_restore = self.extract(
            dst,
            jpeg_block_size=jpeg_block_size,
            wm_size=len(wm)*8,
            refer=wm,
            measures=measures,
            context=ctx,
            savefile=save_kp_img + '.ext.jpg' if save_kp_img else '',
        )

        if pretty:
            print(json.dumps(measures, indent=2 if pretty else 0))

        #
        return measures
