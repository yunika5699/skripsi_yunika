from flask import Flask 
from flask import render_template
import os
import cv2
import numpy as np 
import math
import pickle
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from werkzeug.utils import secure_filename
from flask import request, redirect, url_for, send_from_directory
from keras.layers import GlobalAveragePooling2D
from keras import Sequential
from keras.preprocessing.image import ImageDataGenerator
import efficientnet.keras as efn 
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from PIL import Image
from flask_dropzone import Dropzone


app = Flask(__name__)
basedir = os.path.abspath(os.path.dirname(__file__))
model = pickle.load(open('model.pkl', 'rb'))
sigmaX = 10
labelArray = []
inp_shape = (224,224,3)
feature_extractor = Sequential()
feature_extractor.add(efn.EfficientNetB3(weights='imagenet',include_top=False, input_shape=inp_shape))
feature_extractor.add(GlobalAveragePooling2D())
UPLOAD_FOLDER = '/Users/user/Desktop/projects/skripsi_yunika/uploads'
WRITE_PATH = '/Users/user/Desktop/projects/skripsi_yunika/uploads/filtered'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config.update(
    UPLOADED_PATH=os.path.join(basedir, 'uploads'),
    # Flask-Dropzone config:
    DROPZONE_ALLOWED_FILE_TYPE='image',
    DROPZONE_MAX_FILE_SIZE=25,
    DROPZONE_MAX_FILES=30,
    DROPZONE_UPLOAD_MULTIPLE=True,  # enable upload multiple
)
dropzone = Dropzone(app)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/info")
def info():
    return render_template("info.html")


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        for key, file in request.files.items():
            if key.startswith('file'):
        # if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                img = Image.open(file.stream)
                img = np.array(img)
                img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0,0), sigmaX), -4, 128)
                img = cv2.resize(img, (224,224))
                img = np.array([img])

                print(img.shape)
                features = np.array(feature_extractor.predict(img)[0])

                label = model.predict([features])[0]
                result_message = {
                    0: "Tidak ada gejala DR",
                    1: "Terindikasi DR, segera konsultasi ke Dokter"
                }
                result = result_message[label]
                labelArray.append(result)
                print(labelArray)
                return render_template("Predict.html",results=labelArray)
    return render_template("Predict.html",results=labelArray)

@app.route('/result')
def result():
    results = []
    results = labelArray
    return render_template("result.html", results = results)







@app.route('/show/<filename>')
def uploaded_file(filename):
    sigmaX = 10
    fl = 'http://127.0.0.1:5000/uploads/filtered/' + filename

    read_path = os.path.join(UPLOAD_FOLDER,filename)
    write_path = os.path.join(WRITE_PATH,filename)

    image = cv2.imread(read_path)
    gaussian = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0,0), sigmaX), -4, 128)
    gaussian = cv2.resize(gaussian, (224, 224))
    image = cv2.imwrite(write_path,gaussian)
    return render_template('result.html', filename=fl)


@app.route('/uploads/filtered/<filename>')
def send_file(filename):
    return send_from_directory(WRITE_PATH, filename)



app.run(debug=True)