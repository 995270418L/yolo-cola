'''
    web 可视化测试模型
'''
import json

from flask import Flask
from flask import request, Response

from detect import *

app = Flask(__name__)

@app.route("/ping", methods=['GET', 'POST'])
def hello():
    return "hello, cola"

@app.route('/upload', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_byte = file.read()
        data = get_prediction(img_byte)
        res = {}
        res['data'] = data
        res['code'] = 0
        res['message'] = 'ok'
        print("response: {}".format(res))
        return Response(json.dumps(res), mimetype="application/json")
    else:
        res = {
            'code': 4,
            'message': '请求错误'
        }
        return Response(json.dumps(res), mimetype="application/json")

if __name__ == '__main__':
    app.run(host='0.0.0.0')