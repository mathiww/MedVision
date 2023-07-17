from datetime import timedelta


class Config:
    SECRET_KEY = "secret"
    SESSION_PERMANENT = True
    PERMANENT_SESSION_LIFETIME = timedelta(hours=1)
    SESSION_TYPE = "filesystem"
    SESSION_FILE_DIR = "./app/flask_session"