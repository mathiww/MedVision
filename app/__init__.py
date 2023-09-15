from flask import Flask
from flask_session import Session

from config import Config
from .db import db



def create_app():
    app = Flask(__name__)

    app.config.from_object(Config)
    app.config['SESSION_SQLALCHEMY'] = db

    Session(app)
    db.init_app(app)

    with app.app_context():
        from app.views import main
        app.register_blueprint(main)

    with app.app_context():
        db.create_all()

    return app
