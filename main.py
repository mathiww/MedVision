from models.UniversalClassifier.UniversalClassification import PredictDisease

from flask import Flask, render_template, request, session
from flask_session import Session

from base64 import b64encode
from datetime import timedelta
import os

app = Flask(__name__)

app.config['SECRET_KEY'] = 'aaaaaaaaa'
app.config["SESSION_PERMANENT"] = True
app.config["SESSION_TYPE"] = "filesystem"
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=1)

Session(app)


@app.route('/', methods=["GET"])
def menu():
    return render_template("Menu.html")


@app.route('/', methods=["POST"])
def menu_post():
    file = request.files['file']
    session['images'] = file.read()
    session['class_name'], class_index = PredictDisease(session['images'])

    if class_index == 8:
        return render_template("ErrorModal.html", modal_class_index=class_index)
    
    return render_template("ConfirmModal.html", modal_class_name=session['class_name'], modal_class_index=class_index)


@app.route('/classify', methods=["GET"])
def classify_image():
    class_name, class_image = None, None

    if 'class_name' in session and "images" in session:
        class_name, class_image = session['class_name'], session['images']

    return render_template('Classification.html', class_name=class_name, encoded_image=b64encode(class_image).decode('utf-8'))


if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
