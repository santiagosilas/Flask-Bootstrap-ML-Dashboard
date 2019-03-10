def get_features(request_form):
    count_checkboxs = 0
    for _in in request_form:
        if _in[0:3] in 'checkbox':
            count_checkboxs+=1

    features = ''
    count_caixas = 0

    for _in in request_form:
        if _in[0:3] in 'caixa':
            count_caixas += 1
            features += request_form[_in] + ' '
        if count_caixas == count_checkboxs:
            break

    stripped = features.strip()
    splited = stripped.split() 
    features = list(map(float, splited))
    return features

def get_title(scenario):
    if scenario == '1':
        title = 'Breast Cancer Public Dataset'
    elif scenario == '2':
        title = 'Boston House Prices  Public Dataset'
    else:
        title = 'Diabetes  Public Dataset'

    return title