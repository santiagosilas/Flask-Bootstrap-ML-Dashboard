---------------------
GET http://localhost:5001/predict?model=Model1&features=[0.54, 0.96, 0.73, 0.22]
Response: [{'class': 0, 'prob': 0.0}]

---------------------
GET http://localhost:5001/predict/all?model=Model1&data=[[0.57, 0.77, 0.91, 0.12], [0.47, 0.86, 0.17, 0.79], [0.6, 0.44, 0.75, 0.46]]
Response: [[{'class': 0, 'prob': 0.0}], [{'class': 0, 'prob': 0.0}], [{'class': 0, 'prob': 0.0}]]

