from flask_login import UserMixin

from web import db


class User(UserMixin, db.Model):
    __tablename__ = "Пользователи"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(15))
    email = db.Column(db.String(50), unique=True)
    password = db.Column(db.String(80))
    admin = db.Column(db.Boolean, nullable=False)

    def __init__(self, name, email, password, admin):
        self.name = name
        self.email = email
        self.password = password
        self.admin = admin


class Tips(db.Model):
    __tablename__ = "Советы"
    id = db.Column(db.Integer, primary_key=True)
    index = db.Column(db.String(100))
    mark = db.Column(db.String(30))
    tip = db.Column(db.String(2000))

    def __init__(self, index, mark, tip):
        self.index = index
        self.mark = mark
        self.tip = tip


class Explanations(db.Model):
    __tablename__ = "Пояснения"
    id = db.Column(db.Integer, primary_key=True)
    index = db.Column(db.String(100))
    nation = db.Column(db.String(150))
    explanation = db.Column(db.String(5000))

    def __init__(self, index, nation, explanation):
        self.index = index
        self.nation = nation
        self.explanation = explanation


# Добавить сюда же индексы
class Nation_and_index(db.Model):
    __tablename__ = "Страны"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150))
    pdi = db.Column(db.Integer)
    idv = db.Column(db.Integer)
    mas = db.Column(db.Integer)
    uai = db.Column(db.Integer)
    lto = db.Column(db.Integer)
    ivr = db.Column(db.Integer)

    def __init__(self, name, pdi, idv, mas, uai, lto, ivr):
        self.name = name
        self.pdi = pdi
        self.idv = idv
        self.mas = mas
        self.uai = uai
        self.lto = lto
        self.ivr = ivr


class Select_mark(db.Model):
    __tablename__ = "Больший или меньший показатель индекса"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.Boolean())

    def __init__(self, name):
        self.name = name