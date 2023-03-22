#ffmpeg -i data/train/VID_20200905_215311.mp4 -i data/train/VID_20200914_115125.mp4 -y \
#  -ss 3 -c copy -t 5  -map 0:v -map 0:a -map 1:v -map 1:a \
#  -disposition:v:0 default -disposition:v:1 0 \
#  -disposition:a:0 default -disposition:a:1 0 \
#  data/dst.mp4 &&\
#ffplay -i data/dst.mp4

#ffmpeg -i data/dst1.mp4 \
#-vf "
#drawtext=text='this is an blind watermarked video':fontcolor=white:fontsize=72:box=1:boxcolor=black@0.5:boxborderw=60:x=(w-text_w)/2:y=(h-text_h)/2.7
#" \
#-c:v h264_videotoolbox -c:a copy -b:v 10M \
#data/dst.mp4 -y

#ffmpeg -f hevc -i /Users/jiahao.chen/PycharmProjects/watermark/fifo \
#-c:v copy data/dst2.mp4 -y

#ffmpeg -hide_banner -listen 1 -f hevc -i unix:///Users/jiahao.chen/PycharmProjects/watermark/output.sock \
#  -c:v copy \
#  data/dst.mp4 -y

#ffmpeg -hide_banner -i data/dst1.mp4 \
#  -c:v hevc_videotoolbox -b:v 3M \
#  data/dst.mp4 -y

#ffprobe -i data/wechat.jpg -show_streams -hide_banner -of json

ffmpeg -i data/dst1.mp4 -c:v hevc_videotoolbox -b:v 5M -c:a copy \
  -vf scale=iw/1.5:-1 \
  data/dst.mp4 -y

