from flask import Blueprint, render_template, request
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

def make_dataset(data, window_size=5):  # 수정
    feature_list = []

    for i in range(len(data) - window_size + 1):
        feature_list.append(np.array(data.iloc[i:i + window_size]))

    return np.array(feature_list)

bp = Blueprint('main', __name__, url_prefix='/')


@bp.route('/', methods=('GET', 'POST'))
def index():
    if request.method == 'GET':
        return render_template('detection/detection.html')

    if request.method == 'POST':
        f = (request.files['file'])

        from keras.models import model_from_json
        json_file = open("real64model.json", "r")
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights("real64model_weight.h5")


        s = pd.read_csv(f, header=None)
        s = s.astype(float) / 255
        s = s.dropna(axis=1)  # 데이터에서 NaN이 존재한다면 해당 열을 제거
        s_ = make_dataset(s, 5)
        predictions = loaded_model.predict(s_)
        c = 0

        for j in range(0, s_.shape[0]):
            c += mean_squared_error(s_[j, 0, :], predictions[j, 0, :])

        c = c / s_.shape[0]
        pro = c
        return render_template('detection/detection.html', pro=pro)
