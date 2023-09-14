from datetime import timedelta


class Config:
    MAX_CONTENT_LENGHT = 1024 * 1024 * 8 # 8 MB
    SECRET_KEY = 'secret-key'

    # SQLAlchemy
    SQLALCHEMY_DATABASE_URI = 'sqlite:///database.db'

    # Session
    SESSION_PERMANENT = True
    PERMANENT_SESSION_LIFETIME = timedelta(hours=1)
    SESSION_TYPE = 'sqlalchemy'