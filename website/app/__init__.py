# coding: utf-8
from flask import Flask

app = Flask(__name__)
app.debug = True
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'

from app import views, models, utils