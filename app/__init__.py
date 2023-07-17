from flask import Flask
from flask_session import Session

from config import Config

app = Flask(__name__)
app.config.from_object(Config)

from .views import main
from .db import db

app.config['SESSION_SQLALCHEMY'] = db

Session(app)
app.register_blueprint(main)
