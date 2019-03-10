import os

basedir = os.path.abspath(os.path.dirname(__file__))

DEBUG = True
PORT = 5000
HOST="127.0.0.1"
SQLALCHEMY_ECHO = False
SQLALCHEMY_TRACK_MODIFICATIONS = True
SQLALCHEMY_DATABASE_URI="sqlite:///app.db"
SQLALCHEMY_MIGRATE_REPO = os.path.join(basedir, 'db_repository')
SECRET_KEY='3j4k5h43kj5hj234b5jh34lq25b7k234j5bk2j3b532'