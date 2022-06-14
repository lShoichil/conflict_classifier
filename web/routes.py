from flask import flash, redirect, render_template, request, url_for, Blueprint
from flask_login import (
    current_user,
    login_required,
    login_user,
    logout_user,
)
from itsdangerous import SignatureExpired
from web import bcrypt, db
from .models import Contry
import json
from flask_wtf import FlaskForm
from wtforms import SelectField
from ml import Predictor

main_routes = Blueprint("main", __name__, template_folder="templates")


def add_country():
    with open("web/nation_hafstede.txt", "r", encoding='utf-8') as file:
        nations = list(map(lambda x: x.replace("\n", ""), file.readlines()))
    for nation in nations:
        new_country = Contry(
            name=nation
        )
        db.session.add(new_country)
    db.session.commit()
    return nations


class Form(FlaskForm):
    country1 = SelectField('country1', choices=[])
    country2 = SelectField('country2', choices=[])


@main_routes.route("/", methods=["GET", "POST"])
def home():
    # add_country()
    prediction = ''
    form = Form()
    form.country1.choices = [(country.id, country.name) for country in Contry.query.all()]
    form.country2.choices = [(country.id, country.name) for country in Contry.query.all()]
    if request.method == 'POST':
        model_answer = Predictor()
        user_input = request.form.get('user_input_text')
        nation1 = Contry.query.filter_by(id=request.form.get('country1')).first()
        nation2 = Contry.query.filter_by(id=request.form.get('country2')).first()
        nation = [nation1.name, nation2.name]
        print(nation)
        prediction = model_answer.predict(user_input, nation)
        print(nation1.name)
        print(nation2.name)
        print(prediction)
        print(prediction)

    return render_template("index.html", prediction=prediction, form=form)
