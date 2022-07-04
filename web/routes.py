import csv

from flask import redirect, render_template, url_for, Blueprint, request, flash
from flask_login import (
    current_user,
    login_required,
    login_user,
    logout_user
)

from ml import Predictor
from web import bcrypt, db
from .forms import NationForm, SelectMarkForm
from .models import *
from web import login_manager

main_routes = Blueprint("main", __name__, template_folder="templates")


def add_country_first_try():
    check = Nation_and_index.query.all()
    if not check:
        with open('web/add_first_data/nation_and_index.csv', encoding='utf-8', mode="r") as csvfile:
            nations = csv.DictReader(csvfile)
            for nation in nations:
                new_country = Nation_and_index(
                    name=nation['country'].capitalize(),
                    pdi=nation['pdi'],
                    idv=nation['idv'],
                    mas=nation['mas'],
                    uai=nation['uai'],
                    lto=nation['lto'],
                    ivr=nation['ivr']
                )
                db.session.add(new_country)
            db.session.commit()
            return nations


def add_first_tip():
    check = Tips.query.all()
    if not check:
        with open('web/add_first_data/tips.csv', encoding='utf-8', mode="r") as csvfile:
            tips = csv.DictReader(csvfile)
            for tip in tips:
                new_tip = Tips(
                    index=tip['index'],
                    mark=tip['mark'],
                    tip=tip['tips']
                )
                db.session.add(new_tip)
            db.session.commit()
            return tips


def add_first_explanation():
    check = Explanations.query.all()
    if not check:
        with open('web/add_first_data/exp.csv', encoding='utf-8', mode="r") as csvfile:
            exps = csv.DictReader(csvfile)
            for exp in exps:
                new_exp = Explanations(
                    index=exp['index'],
                    nation=exp['nation'],
                    explanation=exp['explanation']
                )
                db.session.add(new_exp)
            db.session.commit()
            return exps


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


model_answer = Predictor()


@main_routes.route('/', methods=['GET', 'POST'])
def log_reg():
    add_country_first_try()
    add_first_tip()
    add_first_explanation()

    login_email = request.form.get('login_email')
    login_password = request.form.get('login_password')
    login_checkbox = request.form.get('login_checkbox')

    register_name = request.form.get('register_name')
    register_email = request.form.get('register_email')
    register_password = request.form.get('register_password')
    register_checkbox = request.form.get('register_checkbox')

    # print(f'{login_email}\n'
    #       f'{login_password}\n'
    #       f'{login_checkbox}\n'
    #       f'-------------------\n'
    #       f'{register_name}\n'
    #       f'{register_email}\n'
    #       f'{register_password}\n'
    #       f'{register_checkbox}\n')

    user_log = User.query.filter_by(email=login_email).first()
    print(user_log)
    if user_log:

        if bcrypt.check_password_hash(user_log.password, login_password):
            login_user(user_log, remember=bool(login_checkbox))
            return redirect(url_for('main.main_page'))
    # else:
    #     return '<h1>Invalid name or password</h1>'

    user_reg = User.query.filter_by(email=register_email).first()
    if not user_reg and register_checkbox:
        hashed_password = bcrypt.generate_password_hash(register_password)
        new_user = User(
            name=register_name,
            email=register_email,
            password=hashed_password,
            admin=False
        )
        db.session.add(new_user)
        db.session.commit()
        return '<h1>Пользователь создан</h1>'
    # else:
    #     return '<h1>Пользователь с таким email-ом уже существует</h1>'

    return redirect(url_for('main.log_reg'))


@main_routes.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('main.log_reg'))


@main_routes.route('/main_page', methods=["GET", "POST"])
@login_required
def main_page():
    # Ответ от моделей
    prediction = ''

    # Класс в котором хранится результат анализа ситуации
    class Answer:
        toxic = 0
        nation_1 = ''
        nation_2 = ''
        index = ''
        probability = ''
        tips_1 = ''
        tips_2 = ''
        exp_1 = ''
        exp_2 = ''

        def __init__(self, toxic, nation_1, nation_2, index, probability,
                     tips_1, tips_2, exp_1, exp_2):
            self.toxic = toxic
            self.probability = probability
            self.nation_1 = nation_1
            self.nation_2 = nation_2
            self.index = index
            self.tips_1 = tips_1
            self.tips_2 = tips_2
            self.exp_1 = exp_1
            self.exp_2 = exp_2

    form = NationForm()

    # Вывод списка стран в данный момент в selectbox
    form.nation1.choices = [(country.id, country.name) for country in Nation_and_index.query.all()]
    form.nation2.choices = [(country.id, country.name) for country in Nation_and_index.query.all()]

    # Текст пользователя
    user_input = request.form.get('user_input_text')

    # Итоговые две национальности
    nation1 = ''
    nation2 = ''

    if request.method == 'POST':
        check_auto_search_nation = request.form.get('nation_checkbox')
        if check_auto_search_nation:
            check = 0
            flash('Мы проанализировали ситуацию по автоматически найденным национальностям. '
                  'Нажмите на "Показать результаты анализа"')
            prediction = model_answer.predict(user_input)
        else:
            nation1 = Nation_and_index.query.filter_by(id=request.form.get('nation1')).first()
            nation2 = Nation_and_index.query.filter_by(id=request.form.get('nation2')).first()
            if nation1 == nation2:
                flash('Выберите две различные национальности или выберите автоматическое определение')
                # Варианты если много или 1 национальность
            else:
                check = 1
                flash('Мы проанализировали ситуацию по указаным вами национальностям. '
                      'Нажмите на "Показать результаты анализа"')
                # Если он сам выбирает национальности
                index = {"order": ["pdi", "idv", "mas", "uai", "lto", "ivr"],
                         "values": [[nation1.idv, nation1.pdi, nation1.mas, nation1.uai, nation1.lto, nation1.ivr],
                                    [nation2.idv, nation2.pdi, nation2.mas, nation2.uai, nation2.lto, nation2.ivr]]}
                prediction = model_answer.predict(user_input, index)
                print(f'{index}'
                      f'{prediction}')

    # Если пользователь сам определяет
    if prediction != '' and check == 1:

        if prediction['toxic'] == 0.0:
            flash('Ваша ситуация не была определена как конфликтная')
            return render_template("index.html", prediction=prediction, form=form, name=current_user.name,
                                   answer=None, user_input=user_input, check_toxic=0)

        conflict_index = prediction["hofstedefier"]["index"]
        probability = prediction["hofstedefier"]["probability"]

        index_hofstede = {'дистанция власти': 'pdi', 'индивидуализм': 'idv', 'маскулинность': 'mas',
                          'избегание неопределенности': 'uai', 'долгосрочная ориентация': 'lto', 'допущение': 'ivr'}

        short_eng_index = index_hofstede[conflict_index]

        tips_1 = Tips.query.filter_by(index=short_eng_index, mark='high').first()
        tips_2 = Tips.query.filter_by(index=short_eng_index, mark='low').first()

        if short_eng_index == 'pdi':
            if nation1.pdi < nation2.pdi:
                tips_1, tips_2 = tips_2, tips_1
        if short_eng_index == 'idv':
            if nation1.idv < nation2.idv:
                tips_1, tips_2 = tips_2, tips_1
        if short_eng_index == 'mas':
            if nation1.mas < nation2.mas:
                tips_1, tips_2 = tips_2, tips_1
        if short_eng_index == 'pdi':
            if nation1.pdi < nation2.pdi:
                tips_1, tips_2 = tips_2, tips_1
        if short_eng_index == 'uai':
            if nation1.uai < nation2.uai:
                tips_1, tips_2 = tips_2, tips_1
        if short_eng_index == 'lto':
            if nation1.lto < nation2.lto:
                tips_1, tips_2 = tips_2, tips_1
        if short_eng_index == 'ivr':
            if nation1.ivr < nation2.ivr:
                tips_1, tips_2 = tips_2, tips_1

        exp_1 = Explanations.query.filter_by(nation=nation1.name, index=short_eng_index).first()
        exp_2 = Explanations.query.filter_by(nation=nation2.name, index=short_eng_index).first()

        print(f'{prediction["toxic"]}/n'
              f'Национальность: {nation1.name} {nation2.name}/n'
              f'Индекс: {conflict_index} {short_eng_index}/n'
              f'Шанс: {probability}/n'
              f'Советы: {tips_1} {tips_2}/n'
              f'Пояснения: {exp_1} {exp_2}/n')

        answer = Answer(
            toxic=prediction['toxic'],
            probability=probability,
            nation_1=nation1.name,
            nation_2=nation2.name,
            index=conflict_index,
            tips_1=tips_1.tip,
            tips_2=tips_2.tip,
            exp_1=exp_1.explanation,
            exp_2=exp_2.explanation
        )
        return render_template("index.html", prediction=prediction, form=form, name=current_user.name,
                               answer=answer, user_input=user_input, check_toxic=1)

    # Если система сама определяет
    elif prediction != '' and check == 0:
        print(prediction)

        if prediction['toxic'] == 0.0:
            flash('Ваша ситуация не была определена как конфликтная')
            return render_template("index.html", prediction=prediction, form=form, name=current_user.name,
                                   answer=None, user_input=user_input, check_toxic=0)

        if len(prediction['nations']) != 2:
            if len(prediction['nations']) > 2:
                flash('К сожалению мы определили больше чем две национальности в описанной ситуации, попробуйте задать их сами')
            elif len(prediction['nations']) == 1:
                flash('К сожалению мы определили только одну национальность в описанной ситуации, попробуйте задать их сами')
            elif len(prediction['nations']) == 0:
                flash('К сожалению мы не определили присутсвутие каких-либо национальстей в описанной ситуации, попробуйте задать их сами')
            return render_template("index.html", prediction=prediction, form=form, name=current_user.name,
                                   answer=None, user_input=user_input, check_toxic=1)

        conflict_index = prediction["hofstedefier"]["index"]
        index_hofstede = {'дистанция власти': 'pdi', 'индивидуализм': 'idv', 'маскулинность': 'mas',
                          'избегание неопределенности': 'uai', 'долгосрочная ориентация': 'lto', 'допущение': 'ivr'}
        short_eng_index = index_hofstede[conflict_index]

        tips_1 = Tips.query.filter_by(index=short_eng_index, mark='high').first()
        tips_2 = Tips.query.filter_by(index=short_eng_index, mark='low').first()

        nations = list(prediction['nations'])

        print(nations)

        nation1 = Nation_and_index.query.filter_by(name=nations[0].capitalize()).first()
        nation2 = Nation_and_index.query.filter_by(name=nations[1].capitalize()).first()

        print(nation1.name,
              nation2.name)

        if short_eng_index == 'pdi':
            if nation1.pdi < nation2.pdi:
                tips_1, tips_2 = tips_2, tips_1
        if short_eng_index == 'idv':
            if nation1.idv < nation2.idv:
                tips_1, tips_2 = tips_2, tips_1
        if short_eng_index == 'mas':
            if nation1.mas < nation2.mas:
                tips_1, tips_2 = tips_2, tips_1
        if short_eng_index == 'pdi':
            if nation1.pdi < nation2.pdi:
                tips_1, tips_2 = tips_2, tips_1
        if short_eng_index == 'uai':
            if nation1.uai < nation2.uai:
                tips_1, tips_2 = tips_2, tips_1
        if short_eng_index == 'lto':
            if nation1.lto < nation2.lto:
                tips_1, tips_2 = tips_2, tips_1
        if short_eng_index == 'ivr':
            if nation1.ivr < nation2.ivr:
                tips_1, tips_2 = tips_2, tips_1

        exp_1 = Explanations.query.filter_by(nation=nation1.name,
                                             index=short_eng_index).first()
        exp_2 = Explanations.query.filter_by(nation=nation2.name,
                                             index=short_eng_index).first()
        probability = prediction["hofstedefier"]["probability"]

        answer = Answer(
            toxic=prediction['toxic'],
            probability=probability,
            nation_1=nation1.name,
            nation_2=nation2.name,
            index=conflict_index,
            tips_1=tips_1.tip,
            tips_2=tips_2.tip,
            exp_1=exp_1.explanation,
            exp_2=exp_2.explanation
        )

        return render_template("index.html", prediction=prediction, form=form, name=current_user.name,
                               answer=answer, user_input=user_input, check_toxic=1)

    return render_template("index.html", prediction=prediction, form=form, name=current_user.name,
                           answer=None, user_input=user_input, check_toxic=1)


@main_routes.route('/index_guide')
@login_required
def index_guide():
    return render_template("index_guide.html")


@main_routes.route('/admin_page')
# @login_required
def admin_page():
    # УБРАТЬ В ИТОГОВОЙ ВЕРСИИ
    # if current_user.admin:
    #     pass
    # else:
    #     current_user.admin = True

    return render_template("admin_page.html")


# --------------------- Администрирование списка пользователей -----------------------

@main_routes.route('/admin_users', methods=["GET", "POST"])
# @login_required
def admin_users():
    # УБРАТЬ В ИТОГОВОЙ ВЕРСИИ
    # if current_user.admin:
    #     pass
    # else:
    #     current_user.admin = True
    #
    # if current_user.admin:
    #     users = User.query.all()
    # else:
    #     pass

    if not Select_mark.query.all():
        new_true = Select_mark(name=True)
        new_false = Select_mark(name=False)
        db.session.add(new_true)
        db.session.add(new_false)
        db.session.commit()

    form = SelectMarkForm()
    form.mark.choices = [(mark.id, mark.name) for mark in Select_mark.query.all()]

    users = User.query.all()
    return render_template("admin_pages/users.html", users=users, form=form)


@main_routes.route('/insert_users', methods=["POST"])
# @login_required
# Доделать добавление
def insert_users():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']

        check_user = User.query.filter_by(email=email).first()
        if not check_user:
            hashed_password = bcrypt.generate_password_hash(password)
            new_user = User(
                name=name,
                email=email,
                password=hashed_password,
                admin=False
            )
            db.session.add(new_user)
            db.session.commit()
            flash("Новый пользователь добавлен")
        else:
            flash("Такой пользователь уже есть в системе")

        return redirect(url_for('main.admin_users'))


@main_routes.route('/update_users', methods=['GET', 'POST'])
# @login_required
def update_users():
    if request.method == 'POST':
        my_user = User.query.get(request.form.get('id'))

        my_user.name = request.form['name']
        my_user.email = request.form['email']

        if request.form.get('mark') == '1':
            my_user.admin = True
        else:
            my_user.admin = False

        print(my_user.admin)
        db.session.commit()
        flash("Данные успешно обновлены")

        return redirect(url_for('main.admin_users'))


@main_routes.route('/delete_users/<user_id>', methods=['POST', 'GET'])
# @login_required
def delete_users(user_id):
    my_user = User.query.get(user_id)
    db.session.delete(my_user)

    db.session.commit()
    flash("Данные успешно удалены")

    return redirect(url_for("main.admin_users"))


# --------------------- Администрирование советов -----------------------

@main_routes.route('/admin_tips', methods=["GET", "POST"])
# @login_required
def admin_tips():
    # УБРАТЬ В ИТОГОВОЙ ВЕРСИИ
    # if current_user.admin:
    #     pass
    # else:
    #     current_user.admin = True
    #
    # if current_user.admin:
    #     users = Tips.query.all()
    # else:
    #     pass

    tips = Tips.query.all()
    return render_template("admin_pages/tips.html", tips=tips)


@main_routes.route('/insert_tips', methods=["POST"])
# @login_required
def insert_tips():
    if request.method == 'POST':
        index = request.form['index']
        mark = request.form['mark']
        tip = request.form['tip']

        my_tip = Tips(index, mark, tip)
        db.session.add(my_tip)
        db.session.commit()

        flash("Новый совет успешно добавлен")

        return redirect(url_for('main.admin_tips'))


@main_routes.route('/update_tips', methods=['GET', 'POST'])
# @login_required
def update_tips():
    if request.method == 'POST':
        my_tip = Tips.query.get(request.form.get('id'))

        my_tip.index = request.form['index']
        my_tip.mark = request.form['mark']
        my_tip.tip = request.form['tip']

        db.session.commit()
        flash("Данные успешно обновлены")

        return redirect(url_for('main.admin_tips'))


@main_routes.route('/delete_tips/<tip_id>', methods=['POST', 'GET'])
# @login_required
def delete_tips(tip_id):
    my_tip = Tips.query.get(tip_id)
    db.session.delete(my_tip)

    db.session.commit()
    flash("Данные успешно удалены")

    return redirect(url_for("main.admin_tips"))


# --------------------- Администрирование описаний для показателей индекса -----------------------

@main_routes.route('/admin_explanations', methods=["GET", "POST"])
# @login_required
def admin_explanations():
    # УБРАТЬ В ИТОГОВОЙ ВЕРСИИ
    # if current_user.admin:
    #     pass
    # else:
    #     current_user.admin = True
    #
    # if current_user.admin:
    #     users = Tips.query.all()
    # else:
    #     pass
    explanations = Explanations.query.all()
    return render_template("admin_pages/explanation.html", explanations=explanations)


@main_routes.route('/insert_explanations', methods=["POST"])
# @login_required
def insert_explanations():
    if request.method == 'POST':
        index = request.form['index']
        nation = request.form['nation']
        explanation = request.form['explanation']

        my_exp = Explanations(index, nation, explanation)
        db.session.add(my_exp)

        db.session.commit()
        flash("Новое описание успешно добавлено")

        return redirect(url_for('main.admin_explanations'))


@main_routes.route('/update_explanations', methods=['GET', 'POST'])
# @login_required
def update_explanations():
    if request.method == 'POST':
        my_exp = Explanations.query.get(request.form.get('id'))

        my_exp.index = request.form['index']
        my_exp.nation = request.form['nation']
        my_exp.explanation = request.form['explanation']

        db.session.commit()
        flash("Данные успешно обновлены")

        return redirect(url_for('main.admin_explanations'))


@main_routes.route('/delete_explanations/<exp_id>', methods=['POST', 'GET'])
# @login_required
def delete_explanations(exp_id):
    my_exp = Explanations.query.get(exp_id)
    db.session.delete(my_exp)

    db.session.commit()
    flash("Данные успешно удалены")

    return redirect(url_for("main.admin_explanations"))


# --------------------- Администрирование стран и индексов хофстеде -----------------------

@main_routes.route('/admin_nation_and_index', methods=["GET", "POST"])
# @login_required
def admin_nation_and_index():
    # УБРАТЬ В ИТОГОВОЙ ВЕРСИИ
    # if current_user.admin:
    #     pass
    # else:
    #     current_user.admin = True
    #
    # if current_user.admin:
    #     users = Tips.query.all()
    # else:
    #     pass

    nation_and_index = Nation_and_index.query.all()
    return render_template("admin_pages/nation_and_index.html",
                           nation_and_index=nation_and_index)


@main_routes.route('/insert_nation_and_index', methods=["POST"])
# @login_required
def insert_nation_and_index():
    if request.method == 'POST':
        name = request.form['name']
        pdi = request.form['pdi']
        idv = request.form['idv']
        mas = request.form['mas']
        uai = request.form['uai']
        lto = request.form['lto']
        ivr = request.form['ivr']

        new_nation = Nation_and_index(name, pdi, idv, mas,
                                      uai, lto, ivr)
        db.session.add(new_nation)

        db.session.commit()
        flash("Новая страна успешно добавлена")

        return redirect(url_for('main.admin_nation_and_index'))


@main_routes.route('/update_nation_and_index', methods=['GET', 'POST'])
# @login_required
def update_nation_and_index():
    if request.method == 'POST':
        my_nation = Nation_and_index.query.get(request.form.get('id'))
        my_nation.name = request.form['name']
        my_nation.pdi = request.form['pdi']
        my_nation.idv = request.form['idv']
        my_nation.mas = request.form['mas']
        my_nation.uai = request.form['uai']
        my_nation.lto = request.form['lto']
        my_nation.ivr = request.form['ivr']

        db.session.commit()
        flash("Данные успешно обновлены")

        return redirect(url_for('main.admin_nation_and_index'))


@main_routes.route('/delete_nation_and_index/<nation_id>', methods=['POST', 'GET'])
# @login_required
def delete_nation_and_index(nation_id):
    my_nation = Nation_and_index.query.get(nation_id)
    db.session.delete(my_nation)

    db.session.commit()
    flash("Данные успешно удалены")

    return redirect(url_for("main.admin_nation_and_index"))
