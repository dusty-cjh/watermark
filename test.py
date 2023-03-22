import cv2 as cv
import numpy as np
from utils.scale import scale2, iscale2
from models.apis import parse_watermark


wm_restore = np.load('data/wm_bits.npy')
bits_per_second = 8*1
pwd = b'123456'
fps = [25, 1]
while wm_restore.size > 154:
    ret = parse_watermark(wm_restore, b'hel00000000', 24, bits_per_second, fps)
    ret = np.packbits(ret).tobytes()
    print(ret)
    wm_restore = wm_restore[77:]

