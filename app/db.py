from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin

from app import app


db = SQLAlchemy(app)


class User(db.Model, UserMixin):
    id = db.Column('id', db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(20), nullable=False, unique=True)
    email = db.Column(db.String(255), nullable=False, unique=True)
    password = db.Column(db.String(80), nullable=False)
    classifications = db.relationship('Classifications', backref='user')

    def __init__(self, username, email, password):
        self.username = username
        self.email = email
        self.password = password


class Classifications(db.Model):
    id = db.Column('id', db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    name = db.Column(db.String(55), nullable=False)
    image = db.Column(db.Text, nullable=False)
    # mimetype = db.Column(db.Text, nullable=False)

    def __init__(self, user_id, name, image):
        self.user_id = user_id
        self.name = name
        self.image = image
