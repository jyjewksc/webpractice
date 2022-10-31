# from doctest import OutputChecker
# from pytube import Playlist
# from moviepy.editor import *

# import sys
from os import makedirs
from pytube import YouTube


def download(videourl, file_name):
    good = "Download Complete!"
    fail = "Please enter Youtube URL!"
    lnotgood ="The Video length is incorrect!"
    try:
        yt = YouTube(videourl)
        if yt.length > 90 and yt.length < 600:
            progMP4 = yt.streams.filter(progressive=True, file_extension='mp4')   # 設定下載方式及格式
            targetMP4 = progMP4.order_by('resolution').desc().first()  # 由畫質高排到低選畫質最高的來下載
            # targetMP4 = yt.streams.filter(progressive=True, file_extension='mp4')   # 設定下載方式及格式
            mp4_name = f'{file_name}.mp4'  # 給下載影片名
            mp4_path = f'./userfile/{file_name}/mp4'
            makedirs(mp4_path, exist_ok=True)
            targetMP4.download(output_path=mp4_path, filename=mp4_name ) # 下載影片並給輸出路徑    
            validstr = good
        else:
            validstr = lnotgood
    except:
        validstr = fail

    return validstr