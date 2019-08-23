import logging
import os.path

from flask import Flask, request, render_template

logging.basicConfig(format='%(asctime)s %(message)s', filename='logs/ui_logs.log', level=logging.DEBUG)
import util
from database import Database
import os
import urllib.request
import numpy as np
import pandas as pd
import cv2
import urllib


app = Flask(__name__)


@app.route("/test", methods=['GET'])
def test():
    return "This is a test"


@app.route("/templates", methods=['GET'])
def get_templates():
    database = Database()
    database_connection = database.get_connection()
    templates = pd.read_sql(util.GET_TEMPLATE_QUERY, con=database_connection)
    del database
    return render_template("templates.html", memes=templates)


@app.route("/reports", methods=['GET'])
def get_reports():
    database = Database()
    database_connection = database.get_connection()
    template = request.args.get('template')
    records_per_page = int(request.args.get('records_per_page'))
    page_no = int(request.args.get('page_no'))
    if request.args.get('score'):
        score = float(request.args.get('score'))
    else:
        score = 0
    lower_limit = (records_per_page * (page_no-1))
    upper_limit = (records_per_page * page_no)
    memes = pd.read_sql(util.GET_MEMES_FOR_TEMPLATE, params={'template_name': template,
                                                             'lower_limit': lower_limit,
                                                             'upper_limit': upper_limit,
                                                             'score': score},
                        con=database_connection)
    i = 0
    all_memes = []
    meme_list = []
    for key, value in memes.iterrows():
        meme_list.append(value)
        i += 1
        if i % 4 == 0:
            all_memes.append(meme_list)
            meme_list = []
    if len(meme_list) > 0:
        all_memes.append(meme_list)
    del database
    return render_template("report.html", all_memes=all_memes, template=template)


@app.route("/training", methods=['GET'])
def get_training_data():
    template = request.args.get('template')
    template = os.path.basename(template)
    folder_name = os.path.splitext(template)[0]
    file_list = os.listdir("static/training/"+folder_name+"/")
    file_list = ["training/" + folder_name + "/" + file for file in file_list]
    return render_template("training.html", file_list=file_list, template=template)


@app.route("/testing", methods=['GET'])
def get_testing_data():
    template = request.args.get('template')
    template = os.path.basename(template)
    folder_name = os.path.splitext(template)[0]
    file_list = os.listdir("static/testing/" + folder_name + "/")
    file_list = ["testing/" + folder_name + "/" + file for file in file_list]
    return render_template("testing.html", file_list=file_list, template=template)


@app.route("/delete", methods=['GET'])
def delete_training_example():
    file = request.args.get('file')
    os.remove("static/" + file)
    return "True"


@app.route("/unknown", methods=['GET'])
def add_to_unknown():
    url = request.args.get('url')
    resp = urllib.request.urlopen(url)
    img = np.asarray(bytearray(resp.read()), dtype="uint8")
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, dsize=(300, 300))
    a = urllib.parse.urlparse(url)
    file_name = os.path.basename(a.path)
    cv2.imwrite('static/training/Unknown/' + file_name, img)
    return "True"


if __name__ == "__main__":
    app.run(debug=True, port=5003, host="0.0.0.0")

