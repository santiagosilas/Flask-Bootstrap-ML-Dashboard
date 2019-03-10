# exemplo
import requests
from flask import Flask, request, render_template,redirect, url_for, flash

from app import app, models, utils

# In-Memory Database
database = dict()
model_maps = {'1': 'Model1', '2':  'Model2', '3': 'Model3'}

@app.context_processor
def inject_enumerate():
    return dict(enumerate=enumerate)

@app.route('/')
@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/scenario/<scenario>')
def load_scenario(scenario):
    global database
    
    data, columns = models.get_data(scenario)
    
    if scenario not in database.keys():
        database[scenario] = {'data':data, 'columns': columns}

    return render_template(
            'scenario.html', 
            title = utils.get_title(scenario), 
            scenario = scenario, 
            data = data, 
            columns = columns, 
            selected = None)

@app.route('/relatorio')
def relatorio():
    return render_template('relatorio.html')

@app.route('/graficos')
def graficos():
    return render_template('graficos.html')

@app.route('/benchmarks')
def benchmarks():
    return render_template('benchmarks.html')

@app.route('/copy_to_form/<scenario>/<index>')
def copy_to_form(scenario, index):
    global database
    print('keys in database:', database.keys())
    selected = database[scenario]['data'][int(index)]

    return render_template(
        'scenario.html', 
        title = utils.get_title(scenario), 
        scenario = scenario, 
        data = database[scenario]['data'], 
        columns = database[scenario]['columns'], 
        selected = selected)

@app.route('/enviar/exemplo/<scenario>', methods = ['POST'])
def enviar_exemplo(scenario):
    features = utils.get_features(request.form)

    url = 'http://localhost:5001/predict?model={}&features={}'.format(model_maps[scenario], features)
    
    print('url', url)
    
    resultado  = requests.get(url).json()[0]
    return render_template(
        'scenario.html', 
        title = utils.get_title(scenario), 
        scenario = scenario, 
        data = database[scenario]['data'], 
        columns = database[scenario]['columns'], 
        prediction = resultado)
