from .db import User

from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, EmailField, SubmitField, FileField
from wtforms.validators import InputRequired, Length, Email, ValidationError
from flask_wtf.file import FileAllowed


class RegisterForm(FlaskForm):
    username = StringField(validators=[InputRequired(), Length(min=4, max=20)], render_kw={'placeholder': 'Digite aqui o usuário'})
    email = EmailField(validators=[InputRequired(), Email()], render_kw={'placeholder': 'Digite aqui o e-mail'})
    password = PasswordField(validators=[InputRequired(), Length(min=4, max=20)], render_kw={'placeholder': 'Digite aqui a sua senha'})

    submit = SubmitField("Registrar")

    def validate_data(self, username, email):
        username_exist = User.query.filter_by(username=username.data).first()
        email_exist = User.query.filter_by(email=email.data).first()

        if username_exist:
            raise ValidationError('This username is already been used. Try a different one.')
        elif email_exist:
            raise ValidationError('This e-mail is already been used. Try a different one.')


class LoginForm(FlaskForm):
    username = StringField(validators=[InputRequired()], render_kw={'placeholder': 'Usuário'})
    password = PasswordField(validators=[InputRequired()], render_kw={'placeholder': 'Senha'})

    submit = SubmitField("Login")


class ImageForm(FlaskForm):
    image = FileField(validators=[InputRequired(), FileAllowed(upload_set=['png', 'tiff', 'jpeg', 'jpg', 'dicom'], message='Tipo de arquivo não permitido.')])