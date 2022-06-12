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
import torch
from ml.models.Toxic import Model as ToxicModel, predict as ToxicPredict

main_routes = Blueprint("main", __name__, template_folder="templates")


@main_routes.route('/', methods=['GET', 'POST'])
def home():
    prediction = ''
    if request.method == 'POST':
        model = ToxicModel()
        model.state_dict(torch.load("ml/to_upload/toxificator-512.pth", map_location=torch.device('cpu')))
        model.eval()
        user_input = request.form.get('user_input_text')
        prediction = ToxicPredict(model, user_input)
        print(prediction)
    return render_template('index.html', prediction=prediction)
