from flask import Flask
from flask_restful import Api
from resources.predictor import PredictorResource, MLResourceList

app = Flask(__name__)
app.config.from_object('config')

api = Api(app)

# A Generic Resource, to handle any problem, any prediction model
api.add_resource(PredictorResource, '/predict')

api.add_resource(MLResourceList, '/predict/all')

if __name__ == '__main__':
    app.run(debug=True, port=5001)
