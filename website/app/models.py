from sklearn.datasets import load_breast_cancer, load_boston, load_diabetes

# Dictionary-like object, the interesting attributes are: 
#    data:, the data to learn, 
#    target: the classification labels, 
#    target_names: the meaning of the labels, 
#    feature_names: the meaning of the features
ds1 = load_breast_cancer()

ds2 = load_boston()

ds3= load_diabetes()

def get_data(scenario):
    if scenario == '1':
        data = [list(sample) for sample in ds1.data[:10] ]
        columns = [column.capitalize() for column in ds1.feature_names]
    elif scenario == '2':
        data = [list(sample) for sample in ds2.data[:10] ]
        columns = [column.capitalize() for column in ds2.feature_names]
    else:
        data = [list(sample) for sample in ds3.data[:10] ]
        columns = [column.capitalize() for column in ds3.feature_names]

    return data, columns
