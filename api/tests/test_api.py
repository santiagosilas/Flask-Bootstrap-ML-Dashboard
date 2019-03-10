from requests import put, get, post, delete
import numpy as np
import pandas as pd
import os
import json

get_features = lambda i:  [round(x, 2) for x in list(np.random.random(4))]

if __name__ == '__main__':

    features = get_features(4)
    data = [get_features(4) for x in range(3)]
    model = 'Model1'
    
    # GET http://localhost:5001/predict?...}
    url = 'http://localhost:5001/predict?model={}&features={}'.format(model, features)

    print('---------------------')
    print('GET {}'.format(url))
    print('Response:', get(url).json())
    print()
    

    # GET http://localhost:5001/predict/all?...}
    
    url = 'http://localhost:5001/predict/all?model={}&data={}'.format(model, data)
    
    print('---------------------')
    print('GET {}'.format(url))
    print('Response:', get(url).json())
    print()
