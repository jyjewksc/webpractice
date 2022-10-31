#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 11:33:53 2022
相似歌推薦，匯入 user song .csv 版本
@author: jingzhelin
"""
# import sys
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing
# import sys

def find_similar_songs(file_name):
    # 讀資料集 .csv檔，先不設定 index(index_col = False)
    data = pd.read_csv('./finalcsv/after_select_ft_final30s.csv',index_col=False)
    data = data.dropna()
    data = data.drop(columns=['Unnamed: 0','song_name','label'])  # 刪除不要的欄位
    
    # 曲歌單欄位對照 df 最後合併用
    songlist = data.iloc[:,-3:]
    songlist.set_index(['songid'], inplace=True)
    songlist = songlist.drop_duplicates() # 刪除相同欄位
    
    # data 無用欄位處理
    data.set_index(['songid'], inplace=True) # 設定 "song_name" 為 index
    data = data.drop(columns=['videoname','url'])  # 刪除多餘的欄位
    
    # 讀預測歌曲特徵csv
    userdf_3s2 = pd.read_csv(f'./userfile/{file_name}/csv/{file_name}.csv', index_col=False)
    userdf_3s2 = userdf_3s2.dropna()
    userdf_3s2 = userdf_3s2.drop(columns=['Unnamed: 0','song_name','label'])  # 刪除不要的欄位
    userdf_3s2.set_index(['songid'], inplace=True) # 設定 "song_name" 為 index
    
    # data(finalcsv資料) 合併 usercsv 
    data = pd.concat([data, userdf_3s2], axis=0)
    
    # 將相同 “風格+編號”群組，並將所有特徵取平均值，也就是一首歌一個 row
    data = data.groupby(as_index=True, level=0).mean() # 以index 分群
    labels = data.index # 取出 index物件備用
    
    # Scale the data
    data_scaled=preprocessing.scale(data) # 特徵標準化
    
    # Cosine similarity 餘弦相識度模型
    similarity = cosine_similarity(data_scaled) 
    
    # 轉為 df 並改 index、columns 歌名的混淆矩陣
    sim_df_labels = pd.DataFrame(similarity)  # np.array 轉 df
    sim_df_names = sim_df_labels.set_index(labels) # index 換歌名取名稱
    sim_df_names.columns = labels # colums換歌曲名稱

    # 以輸入的歌名取欄欄位，此時格式為 series 再排序相關性，並且排除自己對應的係數＝1
    series = sim_df_names["user"].sort_values(ascending = False) # 相關性高到低
    series = series.drop("user") # 排除自己對應的係數=1
    series = series.iloc[0:10,] # 篩選前五筆
    tmpdf = pd.DataFrame(series, index=series.index) # series -> dataframe
    # 合併
    similar_list = tmpdf.join(songlist)
    
    # 印出標題輸入歌曲名稱和前面相識最高的前五首歌名
    return similar_list

    



