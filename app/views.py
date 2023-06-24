from flask import Blueprint, render_template, request, session

from .models.UniversalClassifier.UniversalClassification import PredictDisease

from base64 import b64encode


main = Blueprint('main', __name__)


@main.route('/', methods=["GET", "POST"])
def menu():
    if request.method == "GET":
        return render_template("Menu.html")


    file = request.files['file']
    session['images'] = file.read()
    session['class_name'], class_index = PredictDisease(session['images'])

    if class_index == 8:
        return render_template("ErrorModal.html", modal_class_index=class_index)
    
    return render_template("ConfirmModal.html", modal_class_name=session['class_name'], modal_class_index=class_index)


@main.route('/classify', methods=["GET"])
def classify_image():
    class_name, class_image = None, None

    if 'class_name' in session and 'images' in session:
        class_name, class_image = session['class_name'], session['images']

    return render_template('Classification.html', class_name=class_name, encoded_image=b64encode(class_image).decode('utf-8'))