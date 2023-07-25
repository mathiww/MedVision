from flask import Blueprint, render_template, session, redirect, jsonify, url_for

from .models import PredictImageType, PredictDisease 

from base64 import b64encode

from flask_login import login_user, LoginManager, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, EmailField, SubmitField, FileField
from wtforms.validators import InputRequired, Length, Email, ValidationError
from flask_wtf.file import FileAllowed

from flask_bcrypt import Bcrypt

from .db import * # Importing app from here


main = Blueprint('main', __name__)

bcrypt = Bcrypt(app)

login_manager = LoginManager(app)
login_manager.login_view = 'login'


def clear_session(keys: list = None):
    if keys:
        for k in keys:
            session.pop(k)
        return
    
    session.clear()

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


class RegisterForm(FlaskForm):
    username = StringField(validators=[InputRequired(), Length(min=4, max=20)], render_kw={'placeholder': 'Username'})
    email = EmailField(validators=[InputRequired(), Email()], render_kw={'placeholder': 'E-mail'})
    password = PasswordField(validators=[InputRequired(), Length(min=4, max=20)], render_kw={'placeholder': 'Password'})

    submit = SubmitField("Resgiter")

    def validate_data(self, username, email):
        username_exist = User.query.filter_by(username=username.data).first()
        email_exist = User.query.filter_by(email=email.data).first()

        if username_exist:
            raise ValidationError('This username is already been used. Try a different one.')
        elif email_exist:
            raise ValidationError('This e-mail is already been used. Try a different one.')


class LoginForm(FlaskForm):
    username = StringField(validators=[InputRequired()], render_kw={'placeholder': 'Username or e-mail'})
    password = PasswordField(validators=[InputRequired()], render_kw={'placeholder': 'Password'})

    submit = SubmitField("Login")


class ImageForm(FlaskForm):
    image = FileField(validators=[InputRequired(), FileAllowed(upload_set=['png', 'tiff', 'jpeg', 'jpg', 'dicom'], message='File type not allowed.')])


@app.errorhandler(404) 
def not_found(e):
  return render_template("404.html")
 

@main.route('/', methods=['GET'])
def home():
    return redirect(url_for('main.login'))


@main.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()

    if form.validate_on_submit():
        form.validate_data(username=form.username, email=form.email)

        hashed_password = bcrypt.generate_password_hash(form.password.data)
        new_user = User(username=form.username.data, email=form.email.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        login_user(new_user)

        return redirect(url_for('main.dashboard'))

    return render_template('Register.html', form=form)


@main.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()

    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()

        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user)
            return redirect(url_for('main.dashboard'))

    return render_template('Login.html', form=form)
     

@main.route('/logout', methods=['GET', 'POST'])
def logout():
    logout_user()
    
    return redirect(url_for('main.login'))


@main.route('/dashboard', methods=['GET'])
@login_required
def dashboard():
    form = ImageForm()
    
    return render_template('Dashboard.html', form=form, classifications=current_user.classifications)

# @main.route('/liver-dashboard', methods=['GET', 'POST'])
# @login_required
# def liver_dashboard():
#     pred_dict = PredictDisease(session['image'], session['class_index'])

#     new_classification = Classifications(
#         user_id=current_user.id,
#         image=b64encode(session['image']).decode('utf-8'),
#         class_name=session['class_name'],
#         prediction=pred_dict,
#     )
#     db.session.add(new_classification)
#     db.session.commit()

#     clear_session(['class_name', 'class_index', 'image'])

#     return render_template('Classification.html')


@main.route('/process-data', methods=['POST'])
@login_required
def process_data():
    form = ImageForm()
    
    if form.validate_on_submit():
        session['image'] = form.image.data.read()
        session['class_name'], session['class_index'] = PredictImageType(session['image'])

        response = {'message': session['class_name'], 'index': str(session['class_index'])}

        return jsonify(response)
    
    errors = form.errors
    return jsonify(errors)


@main.route('/redirect-to-model', methods=['POST'])
@login_required
def redirect_model():    
    pred_dict = PredictDisease(session['image'], session['class_index'])

    new_classification = Classifications(
        user_id=current_user.id,
        image=b64encode(session['image']).decode('utf-8'),
        class_name=session['class_name'],
        prediction=pred_dict,
    )
    db.session.add(new_classification)
    db.session.commit()

    clear_session(['class_name', 'class_index', 'image'])

    return redirect(url_for('main.dashboard'))