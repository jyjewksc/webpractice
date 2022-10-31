import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import os

def pca(file_name):
    pca_jpg_output = f'./userfile/{file_name}/pca_jpg/' 
    if not os.path.isdir(pca_jpg_output):
        os.mkdir(pca_jpg_output)

    select_df = pd.read_csv('./finalcsv/after_select_ft_final30s.csv',index_col=0)
    select_df = select_df.dropna()
    select_df = select_df.drop(columns=['song_name','videoname','url'],axis=1) # 刪除不要的欄位

    userdf_30s2 = pd.read_csv(f'./userfile/{file_name}/csv/{file_name}.csv', index_col=False)
    userdf_30s2 = userdf_30s2.dropna()
    userdf_30s2 = userdf_30s2.drop(columns=['Unnamed: 0','song_name']) # 刪除不要的欄位

    data = pd.concat([select_df, userdf_30s2],ignore_index=True, axis=0)

    data.set_index(['songid'], inplace=True) # 設定 "song_name" 為 index
    data = data.groupby(as_index=True, level=0).mean() # 以index 分群
    data['label'] = data.index.str.split("[0-9]").str.get(0)
    data = data.reset_index()
    data = data.drop(columns=['songid'])

    y = data['label']
    X = data.iloc[:,:-1]

    # #### NORMALIZE X ####
    cols = X.columns
    min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(X)
    X = pd.DataFrame(np_scaled, columns = cols)

    # #### PCA 2 COMPONENTS ####
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(X)
    principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])

    # concatenate with target label
    finalDf = pd.concat([principalDf, y], axis = 1)

    # plot
    plt.figure(figsize = (16, 11)) # 大小

    m = ["o","o","o","o","o","o","o","o","o","o","$\u309a$"] # 標記樣式

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

    # ns.scatterplot(x = "principal component 1", y = "principal component 2",
    #                 data = finalDf, hue = "label", s=120,
    #                 markers=[".","^"], palette=palettes)

    sns.lmplot(data=finalDf, x="principal component 1", y="principal component 2",
               hue="label", palette=colors, fit_reg=False, height=5, aspect=1.5, markers=m, 
               scatter_kws={"s": 40},legend=False)

    # plt.title('PCA on Genres', fontsize = 25)
    # plt.subplots_adjust(top=0.88)
    plt.legend(fontsize='large',loc=2, bbox_to_anchor=(1, 0.5), prop={'size': 10}, markerscale=1)
    plt.tight_layout()
    # plt.xticks(fontsize = 10)
    # plt.yticks(fontsize = 10)
    # plt.xlabel("Principal Component 1", fontsize = 10)
    # plt.ylabel("Principal Component 2", fontsize = 10)
    plt.axis("off")

    plt.savefig(f'./static/pca_jpg/{file_name}.png', dpi=300, transparent=True)
    pca_jpg_path = f'static/pca_jpg/{file_name}.png'

    return pca_jpg_path



