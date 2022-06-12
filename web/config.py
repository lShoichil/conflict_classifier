import os

file_path = os.path.abspath(os.getcwd()) + "/database.db"
SECRET_KEY = 'thisissecret'
SQLALCHEMY_DATABASE_URI = 'sqlite:///' + file_path
