import datetime
import shutil

import cv2
from flask import Blueprint, render_template, request, jsonify, url_for

import torch
import os
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from PIL import Image
import torchvision
from torchvision import transforms, datasets, models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import time

from werkzeug.utils import redirect, secure_filename
import request as req
import pymysql
import cursor

bp = Blueprint('main', __name__, url_prefix='/')


@bp.route('/upload')
def upload():
    return render_template('upload.html')


@bp.route('/result')
def result():
    return render_template('result.html')


@bp.route('/predict', methods=['POST'])
def predict():
    # 필요 함수 정의
    def make_prediction(model, img, threshold):
        model.eval()
        preds = model(img)
        for id in range(len(preds)):
            idx_list = []

            for idx, score in enumerate(preds[id]['scores']):
                if score > threshold:
                    idx_list.append(idx)

            preds[id]['boxes'] = preds[id]['boxes'][idx_list].cpu()  # cpu, gpu 할당 오류
            preds[id]['labels'] = preds[id]['labels'][idx_list].cpu()  # cpu, gpu 할당 오류
            preds[id]['scores'] = preds[id]['scores'][idx_list].cpu()  # cpu, gpu 할당 오류

        return preds

    def get_model_instance_segmentation(num_classes):

        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        return model

    # 파이토치 이미지 읽어오기
    file = request.files['file']
    member_id = request.form.get('member_id') # db용 멤버아이디 받아오기
    img2 = Image.open(file).convert("RGB")

    # 파이토치 모델생성
    data_transform = transforms.Compose([  # transforms.Compose : list 내의 작업을 연달아 할 수 있게 호출하는 클래스
        transforms.ToTensor()  # ToTensor : numpy 이미지에서 torch 이미지로 변경
    ])
    model = get_model_instance_segmentation(26)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model = torch.load('hellopython/model/model_ingre_40_211216.pt', map_location=device)

    # 파이토치 모델 예측하기
    img = Image.open(file).convert("RGB")  # 이미지로 열어주고 RGB형태로 변환
    img = data_transform(img)  # 데이터를 넘파이 이미지에서 토치이미지로 변환

    with torch.no_grad():
        pred = make_prediction(model, [img.to(device)], 0.5)
    result1 = str(pred[0]['labels'][0]).split('(')[1].split(')')[0]

    # result1의 25개 클래스를 5개 클래스로 분류
    if result1 == "1" or "2" or "3" or "4" or "5" or "6":
        result4 = "pasta"
    elif result1 == "7" or "8" or "9" or "10":
        result4 = "deopbap"
    elif result1 == "11" or "12" or "13" or "14" or "15":
        result4 = "bibimbap"
    elif result1 == "16" or "17" or "18" or "19" or "20":
        result4 = "steak"
    else:
        result4 = "waffle"

    # 이미지 로컬에 저장
    timestr = time.strftime("%Y%m%d-%H%M%S")
    local_path = "C:/test/" + result4 + "/" + timestr + ".jpg"
    img2.save(local_path, 'JPEG')

    # 재료 리스트
    ingre_list=["","fusilli","penne","spagetti","tomatoSauce","garlic","shrimp","jeyuk",
              "onion","pa","scramble","kongnamul","mush_muchim","mushroom","pork",
             "sigumchi","pimang","yangpa","aspara","beaf","potato","chocolate",
             "cream","blueberry","dough","strawberry"]
    # 로컬에 저장한 이미지 주소를 데이터베이스에 저장
    print(result4)
    print(ingre_list[int(result1)])
    print(member_id)

    db_ingre_insert(result4, ingre_list[int(result1)], member_id, result4 + "\\" + timestr + ".jpg")

    # 결과를 json 으로 변형
    result_json = jsonify({'label': result4})
    #
    # ## keras, opencv 설치 설치할 것.
    # # 케라스 모델생성
    # keras_model = keras.models.load_model('newFlask/model') # 모델 위치
    #
    # # 케라스 이미지 받아오기
    # file = request.files['file']
    # img = Image.open(file)
    #
    # # 케라스 이미지 전처리
    # # 모델에 맞춰서 새로 전처리 할 것.
    # img = np.array(img)
    # img = cv2.resize(img, (200, 200))
    # img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
    # img = img.reshape(1, 200, 200, 1)
    #
    # # 케라스 모델 예측
    # prediction = keras_model.predict(img)
    # label = jsonify(prediction.tolist())

    # return result_json
    return redirect("http://localhost:8081/web/main.do?result3=" + result4)
    # return render_template('result.html', data=result1)
    # result.html 에서 {{data}} 로 받기

@bp.route('/predict2', methods=['POST'])
def predict2():

    ## keras, opencv 설치 설치할 것.
    # 케라스 모델생성
    #keras_model = keras.models.load_model('newFlask/model/model')  # 모델 위치

    # 케라스 이미지 받아오기
    file = request.files['file']
    img = Image.open(file)

    # 케라스 이미지 전처리
    # 모델에 맞춰서 새로 전처리 할 것.
    img = np.array(img)
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
    print(img.shape)
    img = img.reshape(1, 224, 224, 3)

    # 케라스 모델 예측
    #prediction = keras_model.predict(img)
    #result=np.argmax(prediction, axis=1)[0]
    ###return f"{np.argmax(prediction, axis=1)[0]}"
    #label = jsonify(prediction.tolist())
    #return label
    return redirect("http://localhost:8081/web/upload.do/3")

### db 접속 정보
host = 'project-db-stu.ddns.net'
port = 3307
user_id = 'plating'
user_paswd = '12345'
db_name = 'plating'
charset = 'utf8'  # 데이터베이스서버 문자 인코딩 확인할 것


# sql 문
sql_ingre_insert = """
insert into tbl_ingredient (ingre_name, ingre_content, member_id, ingre_pic)
            values(%s,%s,%s,%s)

"""
sql_plating_insert = """
insert into tbl_plating (member_id, ingre_pic, cook_name)
            values(%s,%s,%s)

"""

#### sql db 쿼리
def db_ingre_insert(ingre_name,ingre_content, member_id, path):
    try:
        db = pymysql.connect(host=host,
                             port=port,
                             user=user_id,
                             passwd=user_paswd,
                             db=db_name,
                             charset=charset)
        cursor = db.cursor()
        cursor.execute("set names utf8")
        cursor.execute(sql_ingre_insert, (ingre_name,ingre_content, member_id, path))  # 인자
        db.commit()
        print('complete save to db')
    finally:
        db.close()


def db_plating_insert(member_id, path, cook_name):
    try:
        db = pymysql.connect(host=host,
                             port=port,
                             user=user_id,
                             passwd=user_paswd,
                             db=db_name,
                             charset=charset)
        cursor = db.cursor()
        cursor.execute("set names utf8")
        cursor.execute(sql_plating_insert, (member_id, path, cook_name))  # 인자
        db.commit()
        print('complete save to db')
    finally:
        db.close()



######## 통신 테스트 #################
@bp.route('/test/<language>')
def test(language):
    data = jsonify({'label': language})
    return data
