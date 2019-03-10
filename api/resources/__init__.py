import os, joblib

ml_models = dict()

M1, M2, M3 = 'Model1', 'Model2', 'Model3'

# Carrega Modelos de Mortalidade Materna
for _file in os.listdir('models/{}'.format(M1)):
    ml_models[_file.split('.')[0]] = joblib.load('models/{}/{}'.format(M1, _file))

# Carrega Modelos de Mortalidade Infantil
for _file in os.listdir('models/{}'.format(M2)):
    ml_models[_file.split('.')[0]] = joblib.load('models/{}/{}'.format(M2, _file))

# Carrega Modelos de Mortalidade Infantil
for _file in os.listdir('models/{}'.format(M3)):
    ml_models[_file.split('.')[0]] = joblib.load('models/{}/{}'.format(M3, _file))