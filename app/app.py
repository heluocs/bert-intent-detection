from flask import Flask,request
from model import IntentDetection

app = Flask(__name__)
model = IntentDetection() 


@app.route('/health')
def health():
    return 'ok'


@app.route('/predict')
def predict():
    query = request.args['query']
    return model.predict(query)


if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0',port=80)
