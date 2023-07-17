from datetime import timedelta


class Config:
    SECRET_KEY = 'secret-key'

    # SQLAlchemy
    SQLALCHEMY_DATABASE_URI = 'sqlite:///database.db'

    # Session
    SESSION_PERMANENT = True
    PERMANENT_SESSION_LIFETIME = timedelta(hours=1)
    SESSION_TYPE = 'sqlalchemy'
    SESSION_FILE_DIR = './app/flask_session'