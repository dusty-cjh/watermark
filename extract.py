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
output_raw_bits = conf.get('output_raw_bits')
ctx = WatermarkContext(**conf['wm_config'])
start_time = time.time()

if infile and outfile:
    wm = extract_video_watermark(
        infile, wm_size=ctx.wm_size,
        bytes_per_second=ctx.bytes_per_second,
        output_raw_bits=output_raw_bits,
    )
    print('watermark extracted, cost time: {}'.format(time.time() - start_time), file=sys.stderr)
    print('wm:', wm)
    sys.exit(0)

# read frame from pipe, output format is h265
print('done!')

