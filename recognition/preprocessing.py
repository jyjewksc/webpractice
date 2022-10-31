#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 22:29:21 2022 

*** update 2022/10/23 ***
1. 修改路徑結構 userfile/{file_name}/
   ├ mp4/{file_name}.mp4 
   ├ wav/{file_name}.wav
   ├ 3s/{file_name}-{}-{}.wav    
   ├ 30s/{file_name}-{}.wav  
   ├ csv/{file_name}.csv (只需要 30s)     

2. 自動選特徵：讀取 after_select_ft_final30s.csv ,取欄位轉出至 userfile
                  
@author: jingzhelin
"""
import pandas as pd
import numpy as np
import librosa
import re
from os import listdir, makedirs

def librosa_feature_op(file_name):
    
    # wav 來源路徑
    # wav_outputpath_3s = f'./userfile/{file_name}/3s'
    wav_outputpath_30s = f'./userfile/{file_name}/30s/'

    # csv 存放路徑   
    csv_outputpath_30s = f'./userfile/{file_name}/csv/'
    makedirs(csv_outputpath_30s, exist_ok=True)
    
    # 自動選特徵，取已挑選的 csv 欄位
    df = pd.read_csv('./finalcsv/after_select_ft_final30s.csv',index_col=0)
    select_df = df.drop(df.columns[-2:], axis=1)
    select_cols = select_df.columns.values
    
    # 可註解掉“特徵字典”內的 key:value，挑出需要轉出的特徵。注意：mfcc ＆ mfcc-delta 為一組，需要一起註解掉。
    # 特徵字典
    feature_dir = {
    0: 'song_name',
    1: 'tempo',
    2: 'total_beats',
    3: 'average_beats',
    4: 'chroma_stft_mean',
    5: 'chroma_stft_std',
    6: 'chroma_stft_var',
    7: 'chroma_cq_mean',
    8: 'chroma_cq_std',
    9: 'chroma_cq_var',
    10: 'chroma_cens_mean',
    11: 'chroma_cens_std',
    12: 'chroma_cens_var',
    13: 'melspectrogram_mean',
    14: 'melspectrogram_std',
    15: 'melspectrogram_var',
        
    16: 'mfcc1_mean',
    17: 'mfcc2_mean',
    18: 'mfcc3_mean',
    19: 'mfcc4_mean',
    20: 'mfcc5_mean',
    21: 'mfcc6_mean',
    22: 'mfcc7_mean',
    23: 'mfcc8_mean',
    24: 'mfcc9_mean',
    25: 'mfcc10_mean',
    26: 'mfcc11_mean',
    27: 'mfcc12_mean',
    28: 'mfcc13_mean',
    29: 'mfcc14_mean',
    30: 'mfcc15_mean',
    31: 'mfcc16_mean',
    32: 'mfcc17_mean',
    33: 'mfcc18_mean',
    34: 'mfcc19_mean',
    35: 'mfcc20_mean',
        
    36: 'mfcc1_std',
    37: 'mfcc2_std',
    38: 'mfcc3_std',
    39: 'mfcc4_std',
    40: 'mfcc5_std',
    41: 'mfcc6_std',
    42: 'mfcc7_std',
    43: 'mfcc8_std',
    44: 'mfcc9_std',
    45: 'mfcc10_std',
    46: 'mfcc11_std',
    47: 'mfcc12_std',
    48: 'mfcc13_std',
    49: 'mfcc14_std',
    50: 'mfcc15_std',
    51: 'mfcc16_std',
    52: 'mfcc17_std',
    53: 'mfcc18_std',
    54: 'mfcc19_std',
    55: 'mfcc20_std',
        
    56: 'mfcc1_var',
    57: 'mfcc2_var',
    58: 'mfcc3_var',
    59: 'mfcc4_var',
    60: 'mfcc5_var',
    61: 'mfcc6_var',
    62: 'mfcc7_var',
    63: 'mfcc8_var',
    64: 'mfcc9_var',
    65: 'mfcc10_var',
    66: 'mfcc11_var',
    67: 'mfcc12_var',
    68: 'mfcc13_var',
    69: 'mfcc14_var',
    70: 'mfcc15_var',
    71: 'mfcc16_var',
    72: 'mfcc17_var',
    73: 'mfcc18_var',
    74: 'mfcc19_var',
    75: 'mfcc20_var',
        
    76: 'mfcc1_delta_mean',
    77: 'mfcc2_delta_mean',
    78: 'mfcc3_delta_mean',
    79: 'mfcc4_delta_mean',
    80: 'mfcc5_delta_mean',
    81: 'mfcc6_delta_mean',
    82: 'mfcc7_delta_mean',
    83: 'mfcc8_delta_mean',
    84: 'mfcc9_delta_mean',
    85: 'mfcc10_delta_mean',
    86: 'mfcc11_delta_mean',
    87: 'mfcc12_delta_mean',
    88: 'mfcc13_delta_mean',
    89: 'mfcc14_delta_mean',
    90: 'mfcc15_delta_mean',
    91: 'mfcc16_delta_mean',
    92: 'mfcc17_delta_mean',
    93: 'mfcc18_delta_mean',
    94: 'mfcc19_delta_mean',
    95: 'mfcc20_delta_mean',
        
    96: 'mfcc1_delta_std',
    97: 'mfcc2_delta_std',
    98: 'mfcc3_delta_std',
    99: 'mfcc4_delta_std',
    100: 'mfcc5_delta_std',
    101: 'mfcc6_delta_std',
    102: 'mfcc7_delta_std',
    103: 'mfcc8_delta_std',
    104: 'mfcc9_delta_std',
    105: 'mfcc10_delta_std',
    106: 'mfcc11_delta_std',
    107: 'mfcc12_delta_std',
    108: 'mfcc13_delta_std',
    109: 'mfcc14_delta_std',
    110: 'mfcc15_delta_std',
    111: 'mfcc16_delta_std',
    112: 'mfcc17_delta_std',
    113: 'mfcc18_delta_std',
    114: 'mfcc19_delta_std',
    115: 'mfcc20_delta_std',
        
    116: 'mfcc1_delta_var',
    117: 'mfcc2_delta_var',
    118: 'mfcc3_delta_var',
    119: 'mfcc4_delta_var',
    120: 'mfcc5_delta_var',
    121: 'mfcc6_delta_var',
    122: 'mfcc7_delta_var',
    123: 'mfcc8_delta_var',
    124: 'mfcc9_delta_var',
    125: 'mfcc10_delta_var',
    126: 'mfcc11_delta_var',
    127: 'mfcc12_delta_var',
    128: 'mfcc13_delta_var',
    129: 'mfcc14_delta_var',
    130: 'mfcc15_delta_var',
    131: 'mfcc16_delta_var',
    132: 'mfcc17_delta_var',
    133: 'mfcc18_delta_var',
    134: 'mfcc19_delta_var',
    135: 'mfcc20_delta_var',
        
    136: 'rmse_mean',
    137: 'rmse_std',
    138: 'rmse_var',
    139: 'cent_mean',
    140: 'cent_std',
    141: 'cent_var',
    142: 'spec_bw_mean',
    143: 'spec_bw_std',
    144: 'spec_bw_var',
    145: 'contrast_mean',
    146: 'contrast_std',
    147: 'contrast_var',
    148: 'rolloff_mean',
    149: 'rolloff_std',
    150: 'rolloff_var',
    151: 'poly_mean',
    152: 'poly_std',
    153: 'poly_var',
    154: 'tonnetz_mean',
    155: 'tonnetz_std',
    156: 'tonnetz_var',
    157: 'zcr_mean',
    158: 'zcr_std',
    159: 'zcr_var',
    160: 'harm_mean',
    161: 'harm_std',
    162: 'harm_var',
    163: 'perc_mean',
    164: 'perc_std',
    165: 'perc_var',
    166: 'frame_mean',
    167: 'frame_std',
    168: 'frame_var',
    169: 'songid',  # file_name
    170: 'label'   # file_name
    }

    # 正規表達初始化
    re_np = re.compile(r'[vsm][ate][rda][\w]{0,1}')   
    re_tempo = re.compile(r'tempo')
    re_total = re.compile(r'total')
    re_average = re.compile(r'average')
    re_chroma_stft = re.compile(r'chroma_stft')
    re_chroma_cq = re.compile(r'chroma_cq')
    re_chroma_cens = re.compile(r'chroma_cens')
    re_melspectrogram = re.compile(r'melspectrogram')
    re_mfcc = re.compile(r'mfcc')
    re_delta = re.compile(r'delta')
    re_rmse = re.compile(r'rmse')
    re_cent = re.compile(r'cent')
    re_spec_bw = re.compile(r'spec_bw')
    re_contrast = re.compile(r'contrast')
    re_rolloff = re.compile(r'rolloff')
    re_poly = re.compile(r'poly')
    re_tonnetz = re.compile(r'tonnetz')
    re_zcr = re.compile(r'zcr')
    re_harm = re.compile(r'harm')
    re_perc = re.compile(r'perc')
    re_frame = re.compile(r'frame')
    
    id = 1
    # 初始化建立 Series 物件清單和算法選擇清單。
    feature_obj_dir = {}
    feature_op_list = []
    for key, var in feature_dir.items():
        if var in select_cols:
            feature_obj_dir[key] = pd.Series(dtype='float32',name=var)
            # 調出需要的字串
            feature_op_list += re_tempo.findall(var)
            feature_op_list += re_total.findall(var)
            feature_op_list += re_average.findall(var)
            feature_op_list += re_chroma_stft.findall(var)
            feature_op_list += re_chroma_cq.findall(var)
            feature_op_list += re_chroma_cens.findall(var)
            feature_op_list += re_melspectrogram.findall(var)
            feature_op_list += re_mfcc.findall(var)
            feature_op_list += re_delta.findall(var)
            feature_op_list += re_rmse.findall(var)
            feature_op_list += re_cent.findall(var)
            feature_op_list += re_spec_bw.findall(var)
            feature_op_list += re_contrast.findall(var)
            feature_op_list += re_rolloff.findall(var)
            feature_op_list += re_poly.findall(var)
            feature_op_list += re_tonnetz.findall(var)
            feature_op_list += re_zcr.findall(var)
            feature_op_list += re_harm.findall(var)
            feature_op_list += re_perc.findall(var)
            feature_op_list += re_frame.findall(var)
        
    feature_op_set = set(feature_op_list) # 轉為集合去重複值
    
    # numpy 算法選擇
    def np_operate(math, value):
        result = float()
        if math == ["mean"]:
            result = np.mean(value)
        elif math == ["std"]:
            result = np.std(value)
        elif math == ["var"]:
            result = np.var(value)
        return result
    
    # 找出路徑資料夾下的所有檔案檔名 ,輸出成 list
    file_data= [f for f in listdir(wav_outputpath_30s)]
    door = True   #避免相同特徵遇到 'tempo'、 'beats' 、 'total' 而轉三次，
    for name in file_data:
        if re.findall(r'.wav+', name) == [".wav"]:  #抓出.wav檔案轉檔。
   
            y, sr = librosa.load(wav_outputpath_30s+name) # 用 librosa讀取檔案
            S = np.abs(librosa.stft(y)) # 傅立葉轉換取振幅
            
            # 有在集合內的特徵就轉換
            if 'tempo' or 'beats' or 'total' in feature_op_set and door == True:
                tempo, beats = librosa.beat.beat_track(y=y, sr=sr) 
                door = False  # 跑過一次就關門，下一輪再打開。
           
            if 'chroma_stft' in feature_op_set:
                chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
   
            if 'chroma_cq' in feature_op_set:
                chroma_cq = librosa.feature.chroma_cqt(y=y, sr=sr)
     
            if 'chroma_cens'in feature_op_set:
                chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
     
            if 'melspectrogram' in feature_op_set:
                melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
      
            if 'rmse' in feature_op_set:
                rmse = librosa.feature.rms(y=y)
       
            if 'cent' in feature_op_set:
                cent = librosa.feature.spectral_centroid(y=y, sr=sr)
           
            if 'spec_bw' in feature_op_set:   
                spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
   
            if 'contrast' in feature_op_set:
                contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
          
            if 'rolloff' in feature_op_set:
                rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
             
            if 'poly' in feature_op_set:
                poly_features = librosa.feature.poly_features(S=S, sr=sr)
            
            if 'tonnetz' in feature_op_set:
                tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
              
            if 'zcr' in feature_op_set:
                zcr = librosa.feature.zero_crossing_rate(y)
              
            if 'harm' in feature_op_set:
                harmonic = librosa.effects.harmonic(y)
               
            if 'perc' in feature_op_set:
                percussive = librosa.effects.percussive(y)
                
            if 'mfcc' in feature_op_set:
                mfcc = librosa.feature.mfcc(y=y, sr=sr)
                mfcc_delta = librosa.feature.delta(mfcc)
               
            if 'frame' in feature_op_set:
                onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
                frames_to_time = librosa.frames_to_time(onset_frames[:20], sr=sr)
            door = False  #跑完一輪關門
 
            for key in feature_obj_dir.keys(): #取出 對應的Key值，取出Series 物件，加入值，選擇np算法 np_operate （名稱， 值）
     
                if key == 0:  # song_name、songid、label
                    feature_obj_dir[key]._set_value(id, name) 
                    continue
                if key == 1:  # tempo
                    feature_obj_dir[key]._set_value(id, tempo)  
                    continue
                if key == 2:  # total beats 
                    feature_obj_dir[key]._set_value(id, np.sum(beats))  
                    continue
                if key == 3: 
                    feature_obj_dir[key]._set_value(id, np.average(beats))
                    continue
                if key == 4 or key == 5 or key == 6:  # chroma stft
                    feature_obj_dir[key]._set_value(id, np_operate(re_np.findall(feature_dir.get(key)), chroma_stft))
                    continue
                if key == 7 or key == 8 or key == 9:  # chroma cq
                    feature_obj_dir[key]._set_value(id, np_operate(re_np.findall(feature_dir.get(key)), chroma_cq))  
                    continue
                if key == 10 or key == 11 or key == 12:  # chroma cens
                    feature_obj_dir[key]._set_value(id, np_operate(re_np.findall(feature_dir.get(key)), chroma_cens))  
                    continue
                if key == 13 or key == 14 or key == 15:  # melspectrogram
                    feature_obj_dir[key]._set_value(id, np_operate(re_np.findall(feature_dir.get(key)), melspectrogram))  
                    continue
                
                if key >= 16 and key < 36: # mfcc_mean 1~20
                    feature_obj_dir[key]._set_value(id, np.mean(mfcc[key-16]))
                    continue
                if key >= 36 and key < 56: # mfcc_std 1~20
                    feature_obj_dir[key]._set_value(id, np.std(mfcc[key-36]))
                    continue
                if key >= 56 and key < 76: # mfcc_var 1~20
                    feature_obj_dir[key]._set_value(id, np.var(mfcc[key-56]))
                    continue
                
                if key >= 76 and key < 96: # mfcc_delta_mean 1~20
                    feature_obj_dir[key]._set_value(id, np.mean(mfcc_delta[key-76]))
                    continue
                if key >= 96 and key < 116: # mfcc_delta_std 1~20
                    feature_obj_dir[key]._set_value(id, np.std(mfcc_delta[key-96]))
                    continue
                if key >= 116 and key < 136: # mfcc_delta_var 1~20
                    feature_obj_dir[key]._set_value(id, np.var(mfcc_delta[key-116]))  
                    continue
                
                if key == 136 or key == 137 or key == 138:  # rmse
                    feature_obj_dir[key]._set_value(id, np_operate(re_np.findall(feature_dir.get(key)), rmse)) 
                    continue
                if key == 139 or key == 140 or key == 141:  # cent
                    feature_obj_dir[key]._set_value(id, np_operate(re_np.findall(feature_dir.get(key)), cent))  
                    continue
                if key == 142 or key == 143 or key == 144:  # spectral bandwidth
                    feature_obj_dir[key]._set_value(id, np_operate(re_np.findall(feature_dir.get(key)), spec_bw))  
                    continue
                if key == 145 or key == 146 or key == 147:  # contrast
                    feature_obj_dir[key]._set_value(id, np_operate(re_np.findall(feature_dir.get(key)), contrast))  
                    continue
                if key == 148 or key == 149 or key == 150:  # rolloff
                    feature_obj_dir[key]._set_value(id, np_operate(re_np.findall(feature_dir.get(key)), rolloff))  
                    continue
                if key == 151 or key == 152 or key == 153:  # poly features
                    feature_obj_dir[key]._set_value(id, np_operate(re_np.findall(feature_dir.get(key)), poly_features))  
                    continue
                if key == 154 or key == 155 or key == 156:  # tonnetz
                    feature_obj_dir[key]._set_value(id, np_operate(re_np.findall(feature_dir.get(key)), tonnetz))  
                    continue
                if key == 157 or key == 158 or key == 159:  # zero crossing rate
                    feature_obj_dir[key]._set_value(id, np_operate(re_np.findall(feature_dir.get(key)), zcr))  
                    continue
                if key == 160 or key == 161 or key == 162:  # harmonic
                    feature_obj_dir[key]._set_value(id, np_operate(re_np.findall(feature_dir.get(key)), harmonic))  
                    continue
                if key == 163 or key == 164 or key == 165:  # percussive
                    feature_obj_dir[key]._set_value(id, np_operate(re_np.findall(feature_dir.get(key)), percussive))  
                    continue
                if key == 166 or key == 167 or key == 168:  # frames
                    feature_obj_dir[key]._set_value(id, np_operate(re_np.findall(feature_dir.get(key)), frames_to_time))  
                    continue
                if key == 169 or key == 170:  #songid、label
                    feature_obj_dir[key]._set_value(id, "user") 
                    continue
            id += 1
            print(name)
            
    # 合併成DF 輸出成 csv
    feature_df = pd.DataFrame()
    for key ,value in feature_obj_dir.items():
        feature_df = pd.concat([feature_df, feature_obj_dir[key]], axis=1)
    feature_df.to_csv(f'{csv_outputpath_30s}/{file_name}.csv')

        