import os, sys
import json
import time
import numpy as np
from models.context import WatermarkContext
from models.apis import embed_video_watermark, extract_video_watermark, test_video_watermark

# read config
if len(sys.argv) == 1:
    conf = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wm_conf.json')
else:
    conf = sys.argv[1]
conf = json.loads(open(conf, 'r').read())
infile, outfile = conf.get('infile'), conf.get('outfile')
ctx = WatermarkContext(**conf['wm_config'])
start_time = time.time()

if infile and outfile:
    embed_video_watermark(
        infile, outfile, ctx.wm_bits,
        bit_rate=conf.get('bit_rate'),
        bytes_per_second=ctx.bytes_per_second,
        output_format=conf.get('output_format'),
    )
    print('watermark embedded, watermark time length: {}, video embedding cost time: {}'.format(
          ctx.estimate_video_watermark_duration(), time.time() - start_time), file=sys.stderr)
    sys.exit(0)

# read frame from pipe, output format is h265
