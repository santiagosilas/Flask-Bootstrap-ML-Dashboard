# datasets para testar o notebook
from sklearn.datasets import load_iris, load_breast_cancer

# módulos úteis
from collections import Counter
from tqdm import tqdm
import time
from datetime import datetime
from IPython.display import display, HTML, Image
import inspect
from scipy import interp
from random import randint
from math import sqrt
import scipy.stats
import os, sys
from collections import defaultdict
from datetime import datetime
import logging

# Para padronização dos dados
from sklearn.preprocessing import StandardScaler

# para visualização e seleção de dados
import pandas as pd

# Para lidar com numpy arrays
import numpy as np

# possui alguns datasets de teste
import seaborn as sb

# datasets disponíveis no sklearn
from sklearn import datasets
import sklearn.metrics

# Feature Ranking with Recursive Feature Elimination (RFE)
from sklearn.feature_selection import RFE

# StratifiedKFold is a variation of K-fold, which returns stratified folds, i.e which creates folds by 
# preserving the same percentage for each target class as in the complete set.
from sklearn.model_selection import StratifiedKFold

# Função para dividir o conjunto em treinamento e teste

# Busca em Grid (para Otimização de hiper-parâmetros do modelo)
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV 

# Classificadores
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.svm import SVC

# Métricas de Desempenho
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report, roc_auc_score, roc_curve
from sklearn.metrics import make_scorer, classification_report, accuracy_score

# Para salvar e carregar objetos serializados
from joblib import dump, load

# Undersampling
from imblearn.under_sampling import ClusterCentroids

# Exibição de Gráficos
from matplotlib import pyplot as plt


'''
Best Parameters in Grid Search for TOP 18 in Entire Dataset (Review this experiment approach)
RF: {'criterion': 'gini', 'max_depth': 10, 'n_estimators': 100}
DT: {'criterion': 'gini', 'splitter': 'best'}
SVM: {'C': 10, 'gamma': 0.1, 'kernel': 'rbf'}
'''

def clf_factory(clf_name):
    if clf_name == 'GaussianNB':
        return GaussianNB()
    elif clf_name == 'SVM':
        return SVC(C=10, gamma=0.1, kernel='rbf', probability=True)
    elif clf_name == 'RandomForest':
        return RandomForestClassifier(criterion='gini', max_depth=10, n_estimators=100)
    elif clf_name == 'DecisionTree':
        return DecisionTreeClassifier(criterion='gini', splitter='best')
    elif clf_name == 'AdaBoost':
        return AdaBoostClassifier()
    elif clf_name == 'MLP':
        return MLPClassifier(alpha=1,  max_iter=10000, tol=0.00000000010, learning_rate_init=0.001, solver='adam', hidden_layer_sizes=(100))

# Just simple debug functions
display_debug = True
def debug(*content):
  ''' shows or not debug messages'''
  if display_debug:
    print('[debug]\t', end='')
    print(content)

def alert_info(content): 
    display(HTML('<div style="color:blue; background-color:#CCE5FF"><spam>{}<spam><div>'.format(content)))

def alert_warning(content): 
    display(HTML('<div style="color:black; background-color:#FF8000"><spam>{}<spam><div>'.format(content)))    
    
def alert_danger(content): 
    display(HTML('<div style="color:lightgray; background-color:#990000"><spam>{}<spam><div>'.format(content)))
    
def text_info(content): 
    display(HTML('<spam style="color:#0066CC;">{}<spam>'.format(content)))

def text_warning(content): 
    display(HTML('<spam style="color:#FF8000;">{}<spam>'.format(content)))    
    
def text_danger(content): 
    display(HTML('<spam style="color:#990000;">{}<spam>'.format(content)))


# Get the average confusion matrix
def mean_confusion_matrix(all_cm):
  ''' get the average confusion matrix'''
  mean = np.zeros((2, 2))
  for cm in all_cm:
    mean += cm
  mean = mean / len(all_cm)
  return mean


# A GridSearch shortcut. Very simple. Not used in this experiment
def optimize_parameters(clf, x, y, parameters):
    ''' get the best parameters for a classifier, by using Grid Search Method'''
    grid_search = GridSearchCV(clf, parameters, cv=10,scoring='accuracy')  #  verbose = 5) n_jobs = n_JobsOnMultiCpuCores
    grid_search.fit(x, y)
    return grid_search.best_params_

# Rank features based on RFE algorithm available in sklearn
def rank_features(x, y, classifier, feature_names):
    ''' 
        Ranking features by RFE Model
        RFE is available in sklearn in the following link: 
        https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html 
    '''
    function_name = inspect.stack()[0][3]
    
    # It is arbitrarily chosen to use a decision tree as a baseline for sorting the attributes of the dataset
    clf = classifier()
    
    # The Recursive Feature Elimination (RFE) algorithm select features by 
    # recursively considering smaller and smaller sets of features
    rfe = RFE(clf, n_features_to_select=1)
    rfe.fit(x, y)

    debug(function_name, rfe.ranking_)
    
    debug(function_name ,"Features sorted by their rank:")
    debug(function_name , sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), feature_names)))
    
    # a list of tuples with ranked features
    # [(0, 'a1'), (0, 'a2'), ...]
    ranked_features = sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), feature_names))
    
    # a list of ranked features
    # ['a1', 'a2', 'a3' ...]
    ranked_features = [tup[1] for tup in ranked_features]
    
    # a list of indexes of ranked features
    ranked_indexes = rfe.ranking_
    
    # ranked_index é indexed from 1
    # ranking_0 é indexed from 0
    ranking_0 = ranked_indexes-1    
    
    return ranking_0, ranked_features    


def balance_by_minority_class(X, y):
    ''' This is a simple undersampling strategy, for better performance in this step'''
    
    # Create a temporary DataFrame object, with X and y numpy arrays
    tmp = pd.DataFrame(data=X)
    tmp['classe'] = y
    
    # Group data by class attribute
    tmp = tmp.reset_index(drop=True)
    g = tmp.groupby('classe')

    # Creates a homogeneous (balanced) subdataset, with the same amount of 
    # examples for classes 0 and 1
    balanced = g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True))
    balanced.classe.value_counts()
    data = balanced.values
    
    # Split data in features matrix and class attribute vector
    X_balanced, y_balanced = data[:,:-1], data[:,-1]
    
    # Return balanced dataset
    return X_balanced, y_balanced



def mean_confidence_interval(data, confidence=0.95):
    '''
      When n>=30, the sampling distribution of the sample mean becomes normal. 
    '''
    function_name = inspect.stack()[0][3]
  
    n = len(data)
    u = np.mean(data)
    sigma = np.std(data)

    # In order to estimate the range of population mean, we define standard error of the mean as follows:
    SE = sigma/sqrt(n)

    interval = scipy.stats.norm.interval(0.95, loc=u, scale=SE)
  
    debug(function_name, 'n:',n,'mean:',u,'+-',u - interval[0], 'sigma:', sigma)
  
    return u, u - interval[0]


def mean_tpr_for_base_fpr(tprs_list, fprs_list, auc_list):
    '''Interpolate tprs
    Args:
        tprs_list: A list of tprs. Each tpr is a list of float numbers
        fprs_list: A list of fprs. Each fpr is a list of float numbers
        auc_list: A list of auc values. each auc is a float number
    Returns:
        mean_tpr: a list o float numbers
        std_tpr: a list o float numbers
        base_fpr: a list o float numbers
        mean_auc: a float number
    '''
    
    # base fpr
    base_fpr = np.linspace(0, 1, 100)
    
    # tpr math interpolation
    tprs = list()
    for __tpr, __fpr,  in zip(tprs_list, fprs_list):
        tprs.append(interp(base_fpr,__fpr, __tpr))
        tprs[-1][0] = 0.0
        
    # mean tpr
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
      
    # std tpr 
    std_tpr = np.std(tprs, axis=0)    
    
    # mean AUC
    mean_auc = sklearn.metrics.auc(base_fpr, mean_tpr)
    
    return mean_tpr, std_tpr, base_fpr, mean_auc


# A function to evaluate a classifier by hold-out (train and test datasets)
def evaluate_holdout(model, X_train, y_train, X_test, y_test):
    # train classifier
    model.fit(X_train, y_train)
                
    # get predictions
    y_predicted = model.predict(X_test)
    probs = model.predict_proba(X_test)

    # compute metrics
    accuracy = accuracy_score(y_test, y_predicted)
    cm = confusion_matrix(y_test, y_predicted)
    tn, fp, fn, tp = confusion_matrix(y_test, y_predicted).ravel()
    f1 = f1_score(y_test, y_predicted, labels = [0, 1])
    auc = roc_auc_score(y_test, probs[:, 1])    
    updated_model = model
    
    # tpr and fpr, to plot roc curve
    fprs_list, tprs_list, threshold = roc_curve(y_test, probs[:, 1])
    
    # return results
    return accuracy, cm, tn, fp, fn, tp, f1, auc, updated_model, tprs_list, fprs_list


def evaluate_cross_validation(model, X, y, seed_cv, num_folds): 
    function_name = inspect.stack()[0][3]
    
    # "Provides train/test indices to split data in train/test sets" (from the sklearn docs)
    k_fold = StratifiedKFold(n_splits=num_folds, random_state=seed_cv, shuffle=True)
    
    # statistics list
    accs, cms, tns, fps, fns, tps, f1s, aucs = [],[],[],[],[],[],[],[]
    
    # best clf: major accuracy
    best_clf = None
    best_accuracy = None
    
    #print('Performs a Cross Validation')
    
    tprs_list_of_lists, fprs_list_of_lists = list(), list()
    for k, (train_index, test_index) in enumerate(k_fold.split(X, y)):
        
        # balance partitions
        #print('balance partitions')
        X_train_part, y_train_part = balance_by_minority_class(X[train_index], y[train_index])
        X_test_part, y_test_part = balance_by_minority_class(X[test_index], y[test_index])

        # model training
        #print('model training')
        model.fit(X_train_part, y_train_part)

        # get predictions
        #print('get predictions')
        y_predicted = model.predict(X_test_part)
        probs = model.predict_proba(X_test_part)

        # compute metrics
        #print('compute metrics')
        accuracy = accuracy_score(y_test_part, y_predicted)
        cm = confusion_matrix(y_test_part, y_predicted)
        tn, fp, fn, tp = confusion_matrix(y_test_part, y_predicted).ravel()
        f1 = f1_score(y_test_part, y_predicted, labels = [0, 1])
        auc = roc_auc_score(y_test_part, probs[:, 1])
        
        # tpr and fpr, to plot roc curve
        fprs_list, tprs_list, _threshold = roc_curve(y_test_part, probs[:, 1])
        
        tprs_list_of_lists.append(tprs_list)
        fprs_list_of_lists.append(fprs_list)
        
        # hold best clf
        if best_clf == None:
            best_clf = model
            best_accuracy = accuracy
        elif accuracy > best_accuracy:
            best_clf = model
            best_accuracy = accuracy
            
        
        # hold results
        accs.append(accuracy)
        cms.append(cm)
        tns.append(tn)
        fps.append(fp)
        fns.append(fn)
        tps.append(tp)
        f1s.append(f1)
        aucs.append(auc)
    
    mean_acc = np.mean(accs)
    mean_cm = mean_confusion_matrix(cms)
    mean_tn = np.mean(tns)
    mean_fp = np.mean(fps)
    mean_fn = np.mean(fns)
    mean_tp = np.mean(tps)
    mean_f1 = np.mean(f1s)
    mean_auc = np.mean(aucs)
    
    
    mean_tpr, std_tpr, base_fpr, _mean_auc = mean_tpr_for_base_fpr(tprs_list_of_lists, fprs_list_of_lists, aucs)

    # return results
    # same return of holdout: accuracy, cm, tn, fp, fn, tp, f1, auc, updated_model, fpr, tpr
    return mean_acc, mean_cm, mean_tn, mean_fp, mean_fn, mean_tp, mean_f1, mean_auc, best_clf, mean_tpr, base_fpr

def save_single_roc_curve(output_path, title, tpr, fpr, auc):
    ''' plot roc curve'''
    # plot no skill
    plt.plot([0, 1], [0, 1], linestyle='--')
  
    # plot the roc curve for the model
    plt.plot(fpr, tpr, marker='.', label='Curva ROC: AUC={0:0.2f}'.format(auc))
  
    plt.xlabel('Especificidade')
    plt.ylabel('Sensitividade')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.grid(True)
    plt.legend(loc="lower right")
    plt.title(title)

    # show the plot
    #plt.show()
    
    # save plot
    plt.savefig('{}/{}.png'.format(output_path, title))
    
    #plt.cla()
    plt.clf()


def save_mean_roc_curve(output_path, title, tprs_list, fprs_list, auc_list):
        '''Saves the average roc curve in a file
        Args:
            title: The title of the graph. It also the name of the png generated file
            tprs_list: A list of tprs. Each tpr is a list of float numbers
            fprs_list: A list of fprs. Each fpr is a list of float numbers
            auc_list: A list of auc values. each auc is a float number
        '''
        
        mean_tpr, std_tpr, base_fpr, mean_auc = mean_tpr_for_base_fpr(tprs_list, fprs_list, auc_list)
        
        # plot each tpr / fpr results
        for tprs, fprs in zip(tprs_list, fprs_list):
            plt.plot(fprs, tprs, lw=1, alpha=0.3)
        
        # upper and lower tpr
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

        # plot graph line
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)
        
        std_auc = np.std(auc_list)
        
        # plot mean auc line
        plt.plot(base_fpr, mean_tpr, color='b',
                 label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),lw=2, alpha=.8)        
        
        # plot grey region
        plt.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2)
        
        plt.grid(True)
        
        # plot another stuffs
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        
        output_roc_graphic = os.path.join(output_path, title + '.png')
        plt.savefig(output_roc_graphic)
        
        #plt.show()
        
        #plt.cla()
        plt.clf()


def create_output_folder(output_path):
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    output_models_path = os.path.join(output_path, 'models')
    if not os.path.exists(output_models_path):
        os.mkdir(output_models_path)
    
    output_graphics_path = os.path.join(output_path, 'graphics')
    if not os.path.exists(output_graphics_path):
        os.mkdir(output_graphics_path)

    
def mkdir_safe(path):
    if not os.path.exists(path):
        os.mkdir(path)


def format_str_list(str_list):
    return str(list(str_list)).replace('[','').replace(']','').replace("'",'').replace(',','\t')


