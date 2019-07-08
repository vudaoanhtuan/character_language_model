import os
from flask import Flask, jsonify, request, json, send_file
from predictor import Predictor

app = Flask(__name__)


@app.route('/next_char', methods=['POST'])
def get_next_char():
    data = request.json

    if 'input' in data:
        res = predictor.predict_next_char(data['input'])
        res = json.dumps(res, ensure_ascii=False)
        return res
    else:
        return "error"

@app.route('/current_word', methods=['POST'])
def get_current_word():
    data = request.json

    if 'input' in data:
        res = predictor.predict_current_word(data['input'])
        res = json.dumps(res, ensure_ascii=False)
        return res
    else:
        return "error"


@app.route('/')
def home():
    return send_file('index.html')

if __name__ == '__main__':
    predictor = Predictor()
    predictor.init_model('weight/model.h5', 'weight/vocab.h5')
    port = int(os.environ.get('PORT', 5000))
    host = 'localhost'
    app.run(host=host, port=port)