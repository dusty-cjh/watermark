import pytest
import cv2 as cv
from models import BlindWatermark
from utils import ImageMeasure


def test_blind_watermark():
    # define src img
    src = cv.imread('../data/flower.jpg')
    wm = cv.imread('../data/wm_gray.png', cv.IMREAD_GRAYSCALE)

    # embed
    measures = {}
    bwm = BlindWatermark(password_wm=1, password_img=1)
    dst = bwm.embed(src, wm, wm_content='../data/wm_gray.png', measures=measures)
    cv.imwrite('../data/dst.jpg', dst, [cv.IMWRITE_JPEG_QUALITY, 50])
    dst = cv.imread('../data/dst.jpg')

    # extract
    wm_restore = bwm.extract(dst, out_wm_name='../data/wm_restore.png')

    # measure
    ber = ImageMeasure.BER(wm > 127, wm_restore > 127)
    psnr = ImageMeasure.PSNR(src, dst)
    print('BER:', ber, 'PSNR bgr:', psnr,measures)
