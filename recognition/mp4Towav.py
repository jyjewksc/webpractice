#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 11:22:26 2022

@author: jingzhelin
"""
from os import makedirs  # 建立全路徑資料夾
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
from pydub.utils import make_chunks

def Video_to_wav(file_name):
    
    # 路徑設定 & 確認資料夾，沒有則建立
    # mp4 路徑 & 轉出 wav 音檔路徑
    mp4_path = f'./userfile/{file_name}/mp4'   # mp4 來源路徑
    wav_outputpath = f'./userfile/{file_name}/wav'     # 給 wav 輸出目錄的上一層目錄
    
    # 如果沒有資料夾建立資料夾
    makedirs(wav_outputpath, exist_ok=True)
    
    # wav 3sor 30s存放路徑
    wav_outputpath_3s = f'./userfile/{file_name}/3s'
    makedirs(wav_outputpath_3s, exist_ok=True)
        
    wav_outputpath_30s = f'./userfile/{file_name}/30s'
    makedirs(wav_outputpath_30s, exist_ok=True)


    #開啟影片檔轉為 WAV
    audioclip = VideoFileClip(f'{mp4_path}/{file_name}.mp4').audio   # 先匯入影檔取出音檔
    audioclip.write_audiofile(f'{wav_outputpath}/{file_name}.wav', fps=22050)  # 將取出的音檔寫入剛給的wav名稱及路經。
    audioclip.close()

    wav = AudioSegment.from_file(f'{wav_outputpath}/{file_name}.wav', "wav") # 開啟wav檔案
    
    # 3秒音檔分段
    size_3s = 3000 # 切割的毫秒數 1s=1000
    chunks_3s = make_chunks(wav, size_3s)  # 將檔案切割為30s一塊
    numstop_3s = len(chunks_3s)-11  # 10段，最後不到 30s 的部分
    for i, chunk3s in enumerate(chunks_3s): 
        if i > 19 and i <= numstop_3s and len(chunk3s) >= 2999: # 不取前 60s及最後不到 30s 的部分
            chunk3s.export(f'{wav_outputpath_3s}/'f'{file_name}-{(i-10)//10}-{i%10}.wav', format="wav")
            # print(f'{wav_outputpath_3s}/'f'{file_name}-{(i-10)//10}-{i%10}.wav') # print test file name
            
    # 30秒音檔分段       
    size_30s = 30000  # 切割的毫秒數 1s=1000
    chunks_30s = make_chunks(wav, size_30s)  # 將檔案切割為3s一塊
    numstop_30s = len(chunks_30s)-2  # 1段，最後不到 30s 的部分
    for j, chunk30s in enumerate(chunks_30s):    # 一段一段的chunk取出來
        if j >= 2 and j <= numstop_30s and len(chunk30s) >= 29999 : # 不取前60s及最後不到30s的部分
            chunk30s.export(f'{wav_outputpath_30s}/'f'{file_name}-{j-1}.wav', format="wav")
            # print(f'{wav_outputpath_3s}/'f'{file_name}-{j-1}.wav')   # print test file name