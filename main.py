import time
import numpy as np
import cv2 as cv
from models.transforms import AdaptiveRegionSelectionWatermark
from models.context import WatermarkContext


# # define
# cap = cv.VideoCapture('data/train/VID_20200914_114958.mp4')
# fps = cap.get(cv.CAP_PROP_FPS)
#
# fourcc = cv.VideoWriter_fourcc(*'XVID')
# width, height = cap.get(cv.CAP_PROP_FRAME_WIDTH), cap.get(cv.CAP_PROP_FRAME_HEIGHT)
# output = cv.VideoWriter('data/dst.mp4', fourcc=fourcc, fps=fps, frameSize=(int(width), int(height)))
#
wm_bits = np.unpackbits(np.array(list(b'hello world, fjoijf 3jfijfri93qgvi3jrg3q9fi 3qfi3j fjijq3f4jq'), dtype=np.uint8))
# i = 0
# while cap.isOpened():
#     # get frame info
#     ret, frame = cap.read()
#     if not ret:
#         print("Can't receive frame (stream end?). Exiting ...")
#         break
#     timestamp = cap.get(cv.CAP_PROP_POS_MSEC)
#
#     # embed watermark
#     ctx = WatermarkContext(wm_bits=bool(wm_bits[i] == 1), wm_strength=3)
#     embeder = AdaptiveRegionSelectionWatermark(ctx)
#     start_time = time.time()
#     frame = frame[:, :, [2, 1, 0]]
#     frame = embeder(frame)
#     frame = frame[:, :, [2, 1, 0]]
#     print(timestamp, 'elapsed:', time.time() - start_time, ctx.errors)
#     output.write(frame)
#     i += 1
#     if i == len(wm_bits):
#         break
#
#     # # show frame
#     # cv.imshow('frame', frame)
#     # if cv.waitKey(1) == ord('q'):
#     #     break
#
# cap.release()
# output.release()
# cv.destroyAllWindows()


# # extract watermark
# cap = cv.VideoCapture('data/dst.mp4')
# wm_size = wm_bits.size
# wm_restore = np.zeros(wm_bits.shape)
# i = 0
# st = time.time()
# while cap.isOpened():
#     # get frame info
#     ret, frame = cap.read()
#     if not ret:
#         print("Can't receive frame (stream end?). Exiting ...")
#         break
#     timestamp = cap.get(cv.CAP_PROP_POS_MSEC)
#
#     # embed watermark
#     ctx = WatermarkContext(wm_strength=3, wm_bits=True)
#     extractor = AdaptiveRegionSelectionWatermark(ctx, is_embed=False)
#     start_time = time.time()
#     frame = frame[:, :, [2, 1, 0]]
#     bit = extractor(frame)
#     wm_restore[i] = bit
#     print(timestamp, 'elapsed:', time.time() - start_time, ctx.errors, bit)
#     i += 1
#     if i == len(wm_bits):
#         break
#
# wm = (wm_restore > 0).astype(np.uint8)
# print((wm == wm_bits).sum() / wm.size)
# print('total cost:', time.time() - st)  # 12.200040102005005


# # using ffmpeg and mac h264 hardware
# import ffmpeg
#
# filename = 'data/dst.mp4'
# probe = ffmpeg.probe(filename)
# video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
# width = int(video_stream['width'])
# height = int(video_stream['height'])
# out, _ = (
#     ffmpeg
#     .input(filename)
#     .output('pipe:', format='rawvideo', pix_fmt='rgb24')
#     .run(capture_stdout=True)
# )
#
# st = time.time()
# i = 0
# wm_restore = np.zeros(wm_bits.shape)
# video = (
#     np
#     .frombuffer(out, np.uint8)
#     .reshape([-1, height, width, 3])
# )
#
# for frame in video:
#     # extract wm
#     ctx = WatermarkContext(wm_strength=3, wm_bits=True)
#     extractor = AdaptiveRegionSelectionWatermark(ctx, is_embed=False)
#     start_time = time.time()
#     bit = extractor(frame)
#     wm_restore[i] = bit
#     i += 1
#     if i == len(wm_bits):
#         break
#
# wm = (wm_restore > 0).astype(np.uint8)
# print((wm == wm_bits).sum() / wm.size)
# print('total cost:', time.time() - st)  # 9.956470966339111

from models.apis import embed_video_watermark, extract_video_watermark, test_video_watermark

infile, outfile = 'data/train/concat1.mp4', 'data/dst.mp4'
st, ctx = time.time(), WatermarkContext()
wm = b'hello world'
bytes_per_second = 0.25
pwd = b'123456'
embed_video_watermark(infile, outfile, wm, bytes_per_second=bytes_per_second, ctx=ctx, password=pwd)
print('embed cost time:', time.time() - st); st = time.time()

wm_restore = extract_video_watermark(
    'data/dst.mp4',
    wm,
    len(wm)*8,
    bytes_per_second=bytes_per_second,
    ctx=ctx,
    output_raw_bits='data/wm_bits.npy',
    password=pwd,
)
print('ext cost time:', time.time() - st, wm_restore, ctx.errors)

# test_video_watermark(infile, wm, )
