from flask import Blueprint, render_template, request, session, redirect

from .models.UniversalClassifier.UniversalClassification import PredictDisease
from base64 import b64encode

from flask_login import login_user, LoginManager, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, EmailField, SubmitField
from wtforms.validators import InputRequired, Length, ValidationError
from flask_bcrypt import Bcrypt

from .db import *

main = Blueprint('main', __name__)

bcrypt = Bcrypt(app)

login_manager = LoginManager(app)
login_manager.login_view = 'login'


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


class RegisterForm(FlaskForm):
    username = StringField(validators=[InputRequired(), Length(min=4, max=20)], render_kw={'placeholder': 'Username'})
    email = EmailField(validators=[InputRequired()], render_kw={'placeholder': 'E-mail'})
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

        return redirect('/dashboard')

    return render_template('Register.html', form=form)


@main.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()

    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()

        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user)
            return redirect('/dashboard')

    return render_template('Login.html', form=form)
     

@main.route('/logout', methods=['GET', 'POST'])
def logout():
    logout_user()
    
    return redirect('/login')

@main.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    if request.method == 'GET':
        return render_template('Dashboard.html', classifications=current_user.classifications)


    file = request.files['file']
    session['image'] = file.read()
    session['class_name'], class_index = PredictDisease(session['image'])

    if class_index == 8:
        return render_template('ErrorModal.html', modal_class_index=class_index)
    
    new_classification = Classifications(
        user_id = current_user.id,
        name = session['class_name'],
        image = b64encode(session['image']).decode('utf-8')
    )
    db.session.add(new_classification)
    db.session.commit()
    
    return render_template('ConfirmModal.html', modal_class_name=session['class_name'], modal_class_index=class_index)


@main.route('/classify', methods=['GET'])
@login_required
def classify_image():

    return render_template('Classification.html', class_name=session['class_name'], encoded_image=b64encode(session['image']).decode('utf-8'))