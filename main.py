from flask import Flask, request, jsonify
from flask_cors import CORS
from collections import Counter
import jieba
import joblib

jieba.initialize()
clf = joblib.load("./models/Bayes_sklearn.pkl")
vsm = joblib.load("./models/vsm.pkl")


app = Flask(__name__)
cors = CORS(app)


def check_spam(text: str):
    data = [text]
    data = [jieba.lcut(x) for x in data]
    data = [Counter(d) for d in data]
    x = vsm.transform(data)
    predicted = clf.predict(x)
    return predicted[0] == 1


@app.route("/predict", methods=["POST"])
def predict():
    req = request.get_json()
    text = req["text"]
    v = check_spam(text)
    return jsonify({"predict": bool(v)})


if __name__ == "__main__":
    app.run()
