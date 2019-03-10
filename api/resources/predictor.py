from flask_restful import request, reqparse, abort, Api, Resource
import numpy as np
import os, ast
import joblib
from resources import ml_models


def predict(features, model_name):
    '''
    A Function to Predict a feature vector
    '''
    n = len(features)
    
    ml_model = ml_models['TOP{:02.0f}'.format(n)]['model']
    
    scaler_name = 'ScalerTOP{:02.0f}'.format(n)
    ml_scaler = ml_models[scaler_name]

    features_list = [float(x) for x in features]

    features_np = np.array(features_list)
    features_vector = features_np.astype(float)

    # escalona o vetor de atributos
    [features_vector] = ml_scaler.transform([features_vector])

    # realiza uma predição
    _class = int(ml_model.predict([features_vector])[0])
    _prob = ml_model.predict_proba([features_vector])[0][1]
        
    output = [{'class': _class, 'prob': _prob}]

    print('output prediction', output)
    return output

# Create an argument parsers

#_parser_list = reqparse.RequestParser()
#_parser_list.add_argument('list')
#_parser_list.add_argument('model')

_get_parser = reqparse.RequestParser()
_get_parser.add_argument('features')
_get_parser.add_argument('model')

_post_parser = reqparse.RequestParser()
_post_parser.add_argument('features', action='append')
_post_parser.add_argument('model')

_post_list_parser = reqparse.RequestParser()
_post_list_parser.add_argument('data', type=list, action='append')
_post_list_parser.add_argument('model')

class PredictorResource(Resource):
    '''
    route /predict
    '''
    #def post(self):
    #
    #    # get data from body
    #    args = _post_parser.parse_args()
    #    features = args['features']
    #    model_name = args['model']
    #
    #    return predict(features, model_name)

    def get(self):
        
        # get data from query string
        args = request.args
        
        features = args['features']
        features = ast.literal_eval(features)
        model_name = args['model']
        output = predict(features, model_name)
        
        return output


class MLResourceList(Resource):
    '''
    route /predict/all
    '''
    def get(self):
        
        # get data from query string
        args = request.args
        
        model_name = args['model']
        _data = args['data']
        
        _data = ast.literal_eval(_data)
        outputs = list()
        for features in _data:
            output = predict(features, model_name)
            outputs.append(output)
        return outputs