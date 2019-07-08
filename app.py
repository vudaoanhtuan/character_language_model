import os
from flask import Flask, jsonify, request, json, send_file
from predictor import Predictor

app = Flask(__name__)


@app.route('/next_char', methods=['POST'])
def get_next_char():
    # return str(type(request.data))
    data = request.json

    if 'input' in data:
        res = predictor.predict_next_char(data['input'])
        res = json.dumps(res, ensure_ascii=False)
        return res
    else:
        return "error"

@app.route('/')
def home():
    return send_file('index.html')

if __name__ == '__main__':
    predictor = Predictor()
    predictor.init_model('param/model.h5', 'param/vocab.h5')
    port = int(os.environ.get('PORT', 5000))
    host = '0.0.0.0'
    app.run(host=host, port=port)