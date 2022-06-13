from flask import flash, redirect, render_template, request, url_for, Blueprint
from flask_login import (
    current_user,
    login_required,
    login_user,
    logout_user,
)
from itsdangerous import SignatureExpired
from web import bcrypt, db
from web.models import User
import json
from ml import Predictor
main_routes = Blueprint("main", __name__, template_folder="templates")


@main_routes.route("/", methods=["GET", "POST"])
def home():
    if request.method == 'POST':
        model_answer = Predictor()
        user_input = request.form.get('user_input_text')
        prediction = json.dumps(model_answer.predict(user_input))
        user_select_nation_1 = request.form.get('user_select_nation_1')
        user_select_nation_2 = request.form.get('user_select_nation_2')
        print (user_select_nation_1)
        print(user_select_nation_2)
        print(prediction)

    return render_template("index.html", prediction=prediction)
