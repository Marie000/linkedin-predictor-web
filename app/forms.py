from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, TextAreaField


class DescriptionForm(FlaskForm):
    description = TextAreaField(label="Job Description: ")
    submit = SubmitField(label="Submit")
