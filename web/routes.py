from flask import redirect, render_template, url_for, Blueprint, request
from flask_login import (
    current_user,
    login_required,
    login_user,
    logout_user
)

from ml import Predictor
from web import bcrypt, db
from .forms import LoginForm, RegisterForm, CountryForm
from .models import User, Country
from web import login_manager

main_routes = Blueprint("main", __name__, template_folder="templates")


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


@main_routes.route('/')
def index():
    return render_template('index.html')


@main_routes.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()

    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user:
            if bcrypt.check_password_hash(user.password, form.password.data):
                login_user(user, remember=form.remember.data)
                return redirect(url_for('main.dashboard'))
        return '<h1>Invalid username or password</h1>'
        # return '<h1>' + form.username.data + ' ' + form.password.data + '</h1>'

    return render_template('login.html', form=form)


@main_routes.route('/signup', methods=['GET', 'POST'])
def signup():
    form = RegisterForm()

    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if not user:
            hashed_password = bcrypt.generate_password_hash(form.password.data)
            new_user = User(
                username=form.username.data,
                email=form.email.data,
                password=hashed_password,
                admin=False
            )
            db.session.add(new_user)
            db.session.commit()
            return '<h1>Пользователь создан</h1>'
        return '<h1>Пользователь с таким email-ом уже существует</h1>'
    return render_template('signup.html', form=form)


@main_routes.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('main.index'))


#
# @main_routes.route('/dashboard')
# @login_required
# def dashboard():
#     return render_template('dashboard.html', name=current_user.username)

@main_routes.route('/dashboard', methods=["GET", "POST"])
@login_required
def dashboard():
    add_country_first_try()
    prediction = ''
    form = CountryForm()
    form.country1.choices = [(country.id, country.name) for country in Country.query.all()]
    form.country2.choices = [(country.id, country.name) for country in Country.query.all()]
    if request.method == 'POST':
        model_answer = Predictor()
        user_input = request.form.get('user_input_text')
        nation1 = Country.query.filter_by(id=request.form.get('country1')).first()
        nation2 = Country.query.filter_by(id=request.form.get('country2')).first()
        nation = [nation1.name, nation2.name]
        print(nation)
        prediction = model_answer.predict(user_input, nation)
        print(nation1.name)
        print(nation2.name)
        print(prediction)
        print(prediction)

    return render_template("index__.html", prediction=prediction, form=form, name=current_user.username)


def add_country_first_try():
    country = Country.query.all()
    if not country:
        with open("web/nation_hafstede.txt", "r", encoding='utf-8') as file:
            nations = list(map(lambda x: x.replace("\n", ""), file.readlines()))
        for nation in nations:
            new_country = Country(
                name=nation
            )
            db.session.add(new_country)
        db.session.commit()
        return nations
