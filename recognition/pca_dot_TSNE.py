import pandas as pd
from sklearn import preprocessing
# from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
# import time
from sklearn.manifold import TSNE
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import os



def tsne(file_name):
    pca_jpg_output = f'./userfile/{file_name}/pca_jpg/' 
    if not os.path.isdir(pca_jpg_output):
        os.mkdir(pca_jpg_output)

    select_df = pd.read_csv('./finalcsv/after_select_ft_final30s.csv',index_col=0)
    select_df = select_df.dropna()
    data = select_df.drop(columns=['song_name','videoname','url'],axis=1) # 刪除不要的欄位

    # 上傳版
    # userdf_30s2 = pd.read_csv(f'./userfile/{file_name}/csv/{file_name}.csv', index_col=False)

    # xen版 
    userdf_30s2 = pd.read_csv(f'./userfile/{file_name}/csv/{file_name}.csv', index_col=False)

    userdf_30s2 = userdf_30s2.dropna()
    userdf_30s2 = userdf_30s2.drop(columns=['Unnamed: 0','song_name']) # 刪除不要的欄位

    data = pd.concat([select_df, userdf_30s2],ignore_index=True, axis=0)

    data.set_index(['songid'], inplace=True) # 設定 "song_name" 為 index
    data = data.groupby(as_index=True, level=0).mean() # 以index 分群
    data['label'] = data.index.str.split("[0-9]").str.get(0)
    data = data.reset_index()
    data = data.drop(columns=['songid'])

    data = data.dropna(axis=1)
    y = data['label']
    X = data.loc[:, data.columns != 'label']
    cols = X.columns
    min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(X)
    X = pd.DataFrame(np_scaled, columns = cols)
    # def TSNE(data):
    tsne = TSNE(n_components=2, init='pca', random_state=501)
    datat_tsne = tsne.fit_transform(X)
    # 归一化
    x_min, x_max = datat_tsne.min(0), datat_tsne.max(0)
    X_norm = (datat_tsne - x_min) / (x_max - x_min)  # 归一化

    # return data
    principalDf = pd.DataFrame(data = X_norm, columns = ['principal component x', 'principal component y'])
    y = pd.DataFrame(y)
    principalDf =  pd.concat([principalDf, y], axis=1)

    # plot
    plt.figure(figsize = (16, 11)) # 大小

    m = ["o","o","o","o","o","o","o","o","o","o","x"] # 標記樣式

    colors={'blues':'#023eff',  # 顏色
            'pop':'#ff7c00',
            'hiphop':'#1ac938',
            'classical':'#a1aaff',
            'disco':'#8b2be2',
            'country':'#9f4800', 
            'rock':'#f14cc1',    
            'metal':'#a3a3a3',
            'jazz':'#ffc400', 
            'reggae':'#00d7ff',
            'user':'#000000'}


    sns.lmplot(data=principalDf, x="principal component x", y="principal component y",
            hue="label", palette=colors, fit_reg=False, height=5, aspect=1.5, markers=m,
            scatter_kws={"s": 40},legend=False)


    plt.legend(fontsize='large',loc=2, bbox_to_anchor=(1, 0.5), prop={'size': 10}, markerscale=1)
    plt.tight_layout()
    plt.axis("off")
    plt.xlabel('TSNE1')
    plt.savefig(f'./static/pca_jpg/{file_name}.png', dpi=300,transparent = False)

    pca_jpg_path = f'static/pca_jpg/{file_name}.png'

    return pca_jpg_path