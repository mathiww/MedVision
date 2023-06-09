from flask import Flask, render_template, request
from models.UniversalClassifier.UniversalClassification import PredictDisease
import os

app = Flask(__name__)
CLASS_NAME = ""


@app.route('/', methods=["GET"])
def menu():
    return render_template("menu.html")


@app.route('/', methods=["POST"])
def menu_post():
    global CLASS_NAME

    file = request.files['file']
    img_bytes = file.read()
    CLASS_NAME = PredictDisease(img_bytes)

    return render_template("menu.html", modal_class_name=CLASS_NAME)


@app.route('/classify', methods=["GET"])
def classify_image():
    global CLASS_NAME

    return render_template('classification.html', class_name=CLASS_NAME)


if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
