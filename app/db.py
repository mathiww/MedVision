from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin

from app import app


db = SQLAlchemy(app)


class User(db.Model, UserMixin):
    id = db.Column('id', db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(20), nullable=False, unique=True)
    email = db.Column(db.String(255), nullable=False, unique=True)
    password = db.Column(db.String(80), nullable=False,)

    def __init__(self, username, email, password):
        self.username = username
        self.email = email
        self.password = password