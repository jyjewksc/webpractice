from fileinput import filename
from flask import redirect
from turtle import down
from flask import Flask, render_template, request, jsonify
from flask_moment import Moment
from pathlib import Path
import uuid
import pandas as pd
import recognition.videodownload as download
import recognition.mp4Towav as wav
import recognition.preprocessing as preprocessing
import recognition.load_model_DL as model
import recognition.recommend_songlist as recommend
import recognition.pca_dot_TSNE as pca
import os

# 需先在music-main下建立mp4、wav兩個資料夾 
app = Flask(__name__)
moment = Moment(app)

UPLOAD_FOLDER = Path(__file__).resolve().parent/'userfile'
print(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def return_img_stream(img_local_path):
    """
    工具函數:
    獲取本地圖片流
    :param img_local_path:文件單張圖片的本地絕對路徑
    :return: 圖片流
    """
    import base64
    img_stream = ''
    with open(img_local_path, 'rb') as img_f:
        img_stream = img_f.read()
        img_stream = base64.b64encode(img_stream).decode()
    return img_stream

@app.route('/', methods=['GET'])
def homepage():
    return render_template('index.html', page_header="UPLOAD MUSIC")

@app.route('/analysis', methods=['GET'])
def analysis():
    return render_template('analysis.html', page_header="UPLOAD MUSIC")

@app.route('/preresult', methods=['POST'])
def validdownload():
    # 獲取url
    text = request.form['dlurl']
    print(text)
    filename = str(uuid.uuid4())
    # 下載mp4
    validstr = download.download(text,filename)
    print(validstr)
    response = jsonify({'filename':filename, 'validstr':validstr})
    return response

@app.route('/result', methods=['POST'])
def result():
    filename = request.form['filename']

    return render_template('result.html', filename=filename)

@app.route('/refresh_result', methods=['POST'])
def refresh_result():
    response = jsonify({'type':"0", 'result':'0'})
    filename = request.form['filename']
    folderpath = f'userfile/{filename}'
    wavpath = f'userfile/{filename}/wav'
    csvpath = f'userfile/{filename}/csv'
    csvfile = f'userfile/{filename}/csv/{filename}.csv'
    pcapath = f'userfile/{filename}/pca_jpg'
    recommendcsvpath = f'userfile/{filename}/{filename}_orginal.csv'
    if os.path.isdir(folderpath): # 判斷有沒有影片
        if os.path.isdir(wavpath): # 判斷wav資料夾是否建立
            if os.path.isdir(csvpath): # 判斷csv資料夾是否建立
                if os.path.isfile(csvfile): # 判斷csv檔案是否建立
                    if os.path.isdir(pcapath): # 判斷pca資料夾是否建立
                        if os.path.isfile(recommendcsvpath):# 判斷recommendcsv資料夾是否建立
                            pass
                        else:
                            # analysis - recommendation
                            result_recommend = recommend.find_similar_songs(filename)
                            result_recommend.to_csv(f'./userfile/{filename}/{filename}_orginal.csv')
                            type = '3'
                            response = jsonify({'type':type})    
                    else:
                        # analysis - PCA
                        result_pca = pca.tsne(filename)
                        type = '2'
                        response = jsonify({'type':type, 'result':result_pca})
                else:
                    pass
            else:
                # preprocessing
                preprocessing.librosa_feature_op(filename)
        else:
            # mp4 to wav
            wav.Video_to_wav(filename)
            # model return numpy.string
            result = model.load_DLmodel(filename)
            type = '1'
            response = jsonify({'type':type, 'result':result})
            
            # 以下改圖版本(app.py跟result.html一起改)
            # resultpath = f'static/genreimg/{result}.png'
            # response = jsonify({'type':type, 'result':result,'resultpath':resultpath})
    else :
        pass
     
    return response

@app.route('/refresh_recommend', methods=['POST'])
def refresh_recommend():
    filename = request.form['filename']
    csvresult = pd.read_csv(f'./userfile/{filename}/{filename}_orginal.csv')
    csvresult = csvresult.reset_index()
    csvresult['index'] = csvresult.index+1
    csvresult = csvresult.drop(['songid','user'],axis=1)
    response = csvresult.to_json(orient="records")

    return response

@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/dataset')
def dataset():
    return render_template('dataset.html')


if __name__ == "__main__":
    # app.run(debug=True)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)