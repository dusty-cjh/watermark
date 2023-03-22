import sys
import time
import platform
import PIL.Image
import ffmpeg
import numpy as np
import cv2 as cv
from utils import shuffle
from .transforms import AdaptiveRegionSelectionWatermark
from .context import WatermarkContext
from .hamming_code import encode, decode, get_encoded_data_index

"""
Performance in IPCs shows that socket is better than pipe for data which large than 10Kb
* https://www.baeldung.com/linux/ipc-performance-comparison
"""


def parse_watermark(wm_restore, wm, wm_prefix_bits, bps, fps, password=b'123456'):
    """Confidence can be estimated by the dividing of max correlativity var and mean var """
    if isinstance(wm, bytes):
        wm_size = len(wm) * 8
    elif isinstance(wm, np.ndarray):
        wm_size = wm.size
    else:
        raise ValueError('wm only support bytes and ndarray')

    # get embedded watermark sequence
    wm = encode(wm).astype(float)
    wm[wm == 0] = -1
    idx = get_encoded_data_index(wm_size)[:wm_prefix_bits]
    wm[idx] *= 10
    wm_bits_index = np.round((np.arange(wm_restore.size) * fps[1] * bps) / fps[0]).astype(int)
    wm_bits_index = wm_bits_index[wm_bits_index < wm.size]
    wm_embed = wm[wm_bits_index]

    # restore bit shifting info
    correlativity = []
    for i in range(wm_embed.size):
        tmp = np.roll(wm_embed, i)
        corr = (tmp * wm_restore[:wm_embed.size]).sum()
        correlativity.append(corr)
    i = np.argmax(correlativity)
    # get wm frame from wm stream
    wm_restore_frame = np.zeros(wm_embed.size)
    tmp = wm_restore[i:i+wm_embed.size]
    wm_restore_frame[:tmp.size] = tmp
    if tmp.size < wm_restore_frame.size:
        wm_restore_frame[-i:] = wm_restore[:i]
    # shrink wm info which was amplified by time factor during embedding
    wm_ext = np.zeros(wm.size)
    wm_ext[wm_bits_index] += wm_restore_frame
    ret = decode(wm_ext > 0, output_format=None)
    ret = shuffle.sort_watermark(ret, password=password, wm_prefix_bits=wm_prefix_bits)
    return ret


def embed_video_watermark(
        input: str, output: str, wm: bytes, bytes_per_second=1,
        bit_rate=None, ctx: WatermarkContext = None, output_format=None, password=b'123456', wm_prefix_bits=24):
    """
    Principle: 1 byte(8bit) per second
    """
    wm = shuffle.shuffle_watermark(wm, password, wm_prefix_bits=wm_prefix_bits)
    bits_per_second = bytes_per_second * 8
    wm_bits = encode(wm)
    probe = ffmpeg.probe(input)
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    width = int(video_stream['width'])
    height = int(video_stream['height'])
    bit_rate = bit_rate or int(video_stream['bit_rate'])
    time_base = [int(x) for x in video_stream['time_base'].split('/')]
    pix_fmt = video_stream['pix_fmt']
    duration_ts = video_stream['duration_ts']
    nb_frames = int(video_stream['nb_frames'])
    avg_frame_rate = [int(x) for x in video_stream['avg_frame_rate'].split('/')]  # timestamp = frame_index * fps (sec)
    input_stream = (
        ffmpeg
        .input(input)   # only for mac
        .output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .run_async(pipe_stdout=True)
    )
    codec = 'hevc_videotoolbox' if platform.platform().startswith('macOS') else 'libx265'
    output_conf = {
        'b:v': bit_rate,
        'c:v': codec,
        # 'q:v': 65,
        'crf': 23,
        'r': video_stream['avg_frame_rate'],
        'f': output_format,
        # 'pass': 2,
    }
    output_stream = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height))
        .output(output, pix_fmt=pix_fmt, **{k: v for k, v in output_conf.items() if v is not None})
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )
    # get timestamp wm bits index list
    wm_bits_index = np.round(
        (np.arange(nb_frames) * avg_frame_rate[1] * bits_per_second) / avg_frame_rate[0]
    ).astype(int)
    assert np.any(wm_bits_index[wm_bits_index >= wm_bits.size]), \
        'video too short, can not extract watermark, bits:{}'.format(wm_bits.size)
    wm_bits_index = wm_bits_index[wm_bits_index < wm_bits.size]
    wm_bits_index = np.repeat(
        wm_bits_index[None, :], np.ceil(nb_frames / wm_bits_index.size), axis=0
    ).flatten()[:nb_frames]
    if ctx:
        ctx.wm_bits = wm_bits
        ctx.wm_bits_index = wm_bits_index
    wm_bits_will_be_embedded = wm_bits[wm_bits_index]

    st = time.time()
    for i, bit in enumerate(wm_bits_will_be_embedded):
        in_bytes = input_stream.stdout.read(width * height * 3)
        if not in_bytes:
            assert wm_bits.size <= i, 'video too short, can not embed bit, frame No.{} < {}'.format(i, wm_bits.size)
        in_frame = (
            np
            .frombuffer(in_bytes, np.uint8)
            .reshape([height, width, 3])
        )

        # embed watermark
        c = WatermarkContext(wm_bits=bool(bit == 1), wm_strength=5)
        embeder = AdaptiveRegionSelectionWatermark(c, inplace=False)
        start_time = time.time()
        out_frame = embeder(in_frame)
        # print('elapsed:', time.time() - start_time); start_time = time.time()
        output_stream.stdin.write(out_frame.tobytes())
        # print('total cost time:', time.time() - st, ', write cost time:', time.time() - start_time); st = time.time()

        i += 1
        # if wm_bits_index[i] == wm_bits.size:
        #     break
    output_stream.stdin.close()
    input_stream.stdout.close()
    input_stream.wait(1)
    from subprocess import TimeoutExpired
    try:
        output_stream.wait(3)
    except TimeoutExpired:
        # output_stream.kill()
        pass


def extract_video_watermark(
        input: str, wm: bytes, wm_size: int, output_format=bytes,
        bytes_per_second=1, ctx: WatermarkContext = None, output_raw_bits=None, wm_prefix_bits=24, password=b'123456'):
    bits_per_second = bytes_per_second * 8
    wm_size = wm_size//4*7    # len(bytes) * 8 * (7/4)
    wm_restore = np.zeros(wm_size)
    probe = ffmpeg.probe(input)
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    width = int(video_stream['width'])
    height = int(video_stream['height'])
    time_base = [int(x) for x in video_stream['time_base'].split('/')]
    pix_fmt = video_stream['pix_fmt']
    nb_frames = int(video_stream['nb_frames'])
    avg_frame_rate = [int(x) for x in
                      video_stream['avg_frame_rate'].split('/')]  # accurate timestamp = frame_index * fps (sec)
    input_stream = (
        ffmpeg
        .input(input)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .run_async(pipe_stdout=True)
    )
    wm_bits_index = np.round(
        (np.arange(nb_frames) * avg_frame_rate[1] * bits_per_second) / avg_frame_rate[0]
    ).astype(int)
    assert np.any(wm_bits_index[wm_bits_index >= wm_size]), \
        'video too short, can not extract watermark, bits:{}'.format(wm_size)
    wm_bits_index = wm_bits_index[wm_bits_index < wm_size]
    wm_bits_index = np.repeat(
        wm_bits_index[None, :], np.ceil(nb_frames / wm_bits_index.size), axis=0
    ).flatten()[:nb_frames]
    if ctx and ctx.wm_bits is not None:
        assert wm_size == ctx.wm_bits.size, 'wm size not match with context: {} != {}'.format(wm_size, ctx.wm_bits.size)
        k = min(ctx.wm_bits_index.size, wm_bits_index.size)
        assert np.all(ctx.wm_bits_index[:k] == wm_bits_index[:k]), 'index selection not match'

    # get wm bit per frame
    wm_bits, i = [], 0
    while True:
        in_bytes = input_stream.stdout.read(width * height * 3)
        if not in_bytes:
            break
        in_frame = (
            np
            .frombuffer(in_bytes, np.uint8)
            .reshape([height, width, 3])
        )

        # extract wm
        c = WatermarkContext(wm_strength=3, wm_bits=True)
        extractor = AdaptiveRegionSelectionWatermark(c, is_embed=False)
        bit = extractor(in_frame)
        # print(i, wm_bits_index.size, wm_restore.size, wm_bits_index[i], wm_restore[wm_bits_index[i]])
        wm_restore[wm_bits_index[i]] += bit
        wm_bits.append(bit)
        i += 1
        # if wm_bits_index[i] == wm_restore.size:
        #     break
    wm_bits = np.array(wm_bits)
    if ctx:
        ctx.wm_bits_restore = wm_restore
        if ctx.wm_bits is not None:
            ctx.errors['BER'] = ((wm_restore > 0) != ctx.wm_bits).sum() / ctx.wm_bits.size
    if output_raw_bits:
        np.save(output_raw_bits, wm_bits)

    wm = parse_watermark(wm_bits, wm, wm_prefix_bits, bps=bits_per_second, fps=avg_frame_rate, password=password)
    # wm = (wm_restore > 0).astype(np.uint8)
    # wm = decode(wm, output_format=output_format)
    input_stream.stdout.close()
    input_stream.wait(1)
    return np.packbits(wm).tobytes()


def test_video_watermark(input: str, wm: bytes, bytes_per_second=1, ctx: WatermarkContext = None):
    """
    Principle: 1 byte(8bit) per second
    """
    bits_per_second = bytes_per_second * 8
    wm_bits = encode(wm)
    probe = ffmpeg.probe(input)
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    width = int(video_stream['width'])
    height = int(video_stream['height'])
    nb_frames = int(video_stream['nb_frames'])
    avg_frame_rate = [int(x) for x in video_stream['avg_frame_rate'].split('/')]  # timestamp = frame_index * fps (sec)
    input_stream = (
        ffmpeg
        .input(input)   # only for mac
        .output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .run_async(pipe_stdout=True)
    )
    # get timestamp wm bits index list
    wm_bits_index = np.round(
        (np.arange(nb_frames) * avg_frame_rate[1] * bits_per_second) / avg_frame_rate[0]
    ).astype(int)
    assert wm_bits_index[-1]+1 >= wm_bits.size, \
        'can not embed wm to video, video too short, {} >= {}'.format(wm_bits_index[-1]+1, wm_bits.size)
    if ctx:
        ctx.wm_bits = wm_bits
        ctx.wm_bits_index = wm_bits_index
    #
    wm_bits_index = np.arange(wm_bits.size+1, dtype=int)

    i = 0
    while True:
        in_bytes = input_stream.stdout.read(width * height * 3)
        if not in_bytes:
            break
        in_frame = (
            np
            .frombuffer(in_bytes, np.uint8)
            .reshape([height, width, 3])
        )

        # embed watermark
        from PIL import Image
        c = WatermarkContext(wm_bits=bool(wm_bits[wm_bits_index[i]] == 1), wm_strength=4)
        embeder = AdaptiveRegionSelectionWatermark(c, inplace=False, best_performance=False)
        start_time = time.time()
        out_frame = embeder(in_frame)

        # extract wm
        extractor = AdaptiveRegionSelectionWatermark(c, is_embed=False, best_performance=False)
        bit = extractor(out_frame)
        if int(bit > 0) != int(wm_bits[wm_bits_index[i]]):
            print(c.errors, file=sys.stderr)
            Image.fromarray(in_frame).save('data/dst3.jpg')

        i += 1
        if wm_bits_index[i] == wm_bits.size:
            break
    assert wm_bits_index[i] == wm_bits.size, \
        'video too short, wm embedding not complete, {} < {}'.format(wm_bits_index[i], wm_bits.size)

    input_stream.stdout.close()
    input_stream.wait(1)

