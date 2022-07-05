import pandas as pd
import numpy as np
import pickle

import logging
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
import process_data
import load_data
import yaml

params = yaml.safe_load(open('config.yaml'))[0]

list_names = params['load_data']['list_names']
path = params['load_data']['path']
limit_loads = params['load_data']['limit_loads']

n_size = params['convert']['n_size']
new_path = params['convert']['new_path']

test_size = params['train']['test_size']
random_state = params['train']['random_state']
key_load = params['train']['key_load']
path_model = params['train']['path_model']

logging.basicConfig(format = '%(levelname)s:%(message)s', level=logging.INFO)

def get_results(y_test, y_pred):
    
    """Form dict with the main metrics of the model
    :param y_test:
    :param y_test:
    :return : dict"""
   
    results = dict()
    results['rec'] = recall_score(y_test, y_pred, average = 'micro').round(3)
    results['prec'] = precision_score(y_test, y_pred, average = 'micro').round(3)
    results['acc'] = accuracy_score(y_test, y_pred).round(3)
    results['f1']  = f1_score(y_test, y_pred, average = 'micro').round(3)
    
    return results

def fit_model(X,y,test_size,random_state, path_model):
    
    """Training model for face recognition:
    :param X: embedings
    :param y: target"""

    logging.info('Training model')
    X_train,X_test,y_train,y_test = train_test_split(X, 
                                                     y,
                                                     test_size = test_size,
                                                     stratify = y,
                                                     random_state = random_state)

    clf = LogisticRegression()
    clf.fit(X_train,y_train)
    # write model to the file
    with open(path_model, 'wb') as f:
        pickle.dump(clf, f)

    y_pred = clf.predict(X_test)
    print(get_results(y_test,y_pred))

def get_data(path:str):
    embedings = pd.read_csv(path+'/embendings.csv', sep = ',',index_col= 0)
    target = np.array(pd.read_csv(path+ '/target.csv',index_col= 0))
    target = target.reshape(target.shape[0])
    dict_actors = pd.read_csv(path+'/dict_actors.csv',index_col= 1)
    return embedings, target, dict_actors


if __name__ == "__main__":
    if key_load == True:
        #download data
        load_data.request_download(path, list_names, limit_loads)
        #convert images 
        load_data.convert_im(list_names, n_size, path)
        
        #get embandings
        prda = process_data.GetEmbedings(path, list_names, new_path)
        prda.upload_to()

    embedings, target, dict_actors = get_data(new_path)

    fit_model(embedings,target,test_size,random_state,path_model)