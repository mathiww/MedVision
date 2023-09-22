from flask import current_app, Blueprint, render_template, session, redirect, jsonify, url_for

from .models import PredictImageType, PredictDisease

from base64 import b64encode

from flask_login import login_user, LoginManager, login_required, logout_user, current_user
from flask_bcrypt import Bcrypt

from .forms import LoginForm, RegisterForm, ImageForm
from .db import *

main = Blueprint('main', __name__)

bcrypt = Bcrypt(current_app)

login_manager = LoginManager(current_app)
login_manager.login_view = 'main.login'


def clear_session(*args):
    if args:
        for arg in args:
            session.pop(arg)
        return
    
    session.clear()

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))
 

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
     

@main.route('/logout', methods=['GET'])
@login_required
def logout():
    logout_user()    
    return redirect(url_for('main.login'))


@main.route('/dashboard', methods=['GET'])
@login_required
def dashboard():
    form = ImageForm()
    
    predictions = Classifications.query.filter_by(user_id=int(current_user.id)).order_by(Classifications.id.desc()).limit(5).all()
    session['num_tr'] = 1

    return render_template('Dashboard.html', form=form, classifications=predictions)


@main.route('/process-data', methods=['POST'])
@login_required
def process_data():
    form = ImageForm()
    
    if form.validate_on_submit():
        session['image'] = form.image.data.read()
        session['class_name'], session['class_index'] = PredictImageType(session['image'])

        response = {'message': session['class_name'], 'index': str(session['class_index']), 'image': str(b64encode(session['image']).decode('utf-8'))}

        return jsonify(response)
    
    errors = form.errors
    return jsonify(errors)


@main.route('/redirect-to-model', methods=['POST'])
@login_required
def redirect_model():    
    pred_dict = PredictDisease(session['image'], session['class_index'])

    if Classifications.query.count() >= 100:
        oldest_record = Classifications.query.order_by(Classifications.timestamp).first()

        db.session.delete(oldest_record)
        db.session.commit()

    new_classification = Classifications(
        user_id=current_user.id,
        image=b64encode(session['image']).decode('utf-8'),
        class_name=session['class_name'],
        prediction=pred_dict,
    )
    db.session.add(new_classification)
    db.session.commit()

    clear_session('class_name', 'class_index', 'image')

    return redirect(url_for('main.dashboard'))


@main.get('/load-more')
def load_more():

    predictions = Classifications.query.filter_by(user_id=int(current_user.id)).order_by(Classifications.id.desc()).offset(5 + session['num_tr']).limit(5).all()

    if len(predictions) != 0:
        session['num_tr'] += 5
    else:
        return ''

    return render_template('LoadMore.html', classifications=predictions)


@main.errorhandler(404) 
def not_found(e):
  return render_template("404.html")


@main.errorhandler(405) 
def not_allowed(e):
  return render_template("404.html")