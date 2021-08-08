import os
from flask import Flask, flash, request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename #ファイル名に危険なコマンドなどが含まれていないかチェック
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model #ニューラルネットワークのモデル定義に必要
import sys
import numpy as np
from PIL import Image

# 画像データパラメータ
classes = ['dyed rose', 'not rose', 'rose beige', 'rose bicolor', 'rose brown', 'rose green', 'rose orange', 'rose pink', 'rose purple', 'rose red', 'rose white', 'rose yellow', ]
num_classes = len(classes)
image_size = 50

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__) #アプリケーションをFlaskインスタンスとして初期化
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER #アプリケーションの設定

def allowed_file(filename):
    # ファイル名にピリオドが入っている、ファイル名のピリオド以降の拡張子が正しいか
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('ファイルがありません')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('ファイルがありません')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename)) #ファイルの保存
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename) #ファイルパスの設定

            # モデルのロード
            model = load_model('./rose_model.h5')

            # imageの定義
            image = Image.open(filepath) # python predict.py filename とすると、filenameは２番目の引数のため、[1]とする
            image = image.convert('RGB') # 3色のRGBに変換
            image = image.resize((image_size, image_size))
            data = np.asarray(image)
            X = []
            X.append(data)
            X = np.array(X)

            result = model.predict([X])[0]
            predicted = result.argmax() #１番目に大きい配列を返す
            predicted2 = result.argsort()[-2] #２番目に大きい配列を返す
            predicted3 = result.argsort()[-3] #３番目に大きい配列を返す
            percentage = float(result[predicted] * 100)
            percentage2 = float(result[predicted2] * 100)
            percentage3 = float(result[predicted3] * 100)
            resultmsg = "ラベル1： " + classes[predicted] + ", 確率："+ str(percentage) + " %|" "ラベル2： " + classes[predicted2] + ", 確率："+ str(percentage2) + " %|" "ラベル3： " + classes[predicted3] + ", 確率："+ str(percentage3) + " %|"
            return render_template('answer.html', resultmsg=resultmsg, filepath=filepath)

    return render_template('index.html')

# アップロード後の処理
from flask import send_from_directory

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

# if __name__ == "__main__":
#    app.run()