"""
@author : Sathish Sampath(ss.sathishsampath@gmail.com)

"""


#%% Imports
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from helpers import   nn_arch,nn_reg,ImportanceSelect
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV


if __name__ == '__main__':
    out = './RF/'
    
    np.random.seed(0)
    digits = pd.read_hdf('./BASE/datasets.hdf','digits')
    digitsX = digits.drop('Class',1).copy().values
    digitsY = digits['Class'].copy().values
    
    biodeg = pd.read_hdf('./BASE/datasets.hdf','biodeg')        
    biodegX = biodeg.drop('Class',1).copy().values
    biodegY = biodeg['Class'].copy().values
    
    
    biodegX = StandardScaler().fit_transform(biodegX)
    digitsX= StandardScaler().fit_transform(digitsX)
    
    clusters =  [2,5,10,15,20,25,30,35,40]
    dims = [2,5,10,15,20,25,30,35,40,45,50,55,60]
    
    dimsb = [2,5,7,10,15,20,25,30,35]
    
    #%% data for 1
    
    rfc = RandomForestClassifier(n_estimators=100,class_weight='balanced',random_state=5,n_jobs=7)
    fs_biodeg = rfc.fit(biodegX,biodegY).feature_importances_ 
    fs_digits = rfc.fit(digitsX,digitsY).feature_importances_ 
    
    tmp = pd.Series(np.sort(fs_biodeg)[::-1])
    tmp.to_csv(out+'biodeg scree.csv')
    
    tmp = pd.Series(np.sort(fs_digits)[::-1])
    tmp.to_csv(out+'digits scree.csv')
    
    #%% Data for 2
    filtr = ImportanceSelect(rfc)
    grid ={'filter__n':dimsb,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
    mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
    pipe = Pipeline([('filter',filtr),('NN',mlp)])
    gs = GridSearchCV(pipe,grid,verbose=10,cv=5)
    
    gs.fit(biodegX,biodegY)
    tmp = pd.DataFrame(gs.cv_results_)
    tmp.to_csv(out+'Biodeg dim red.csv')
    
    
    grid ={'filter__n':dims,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}  
    mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
    pipe = Pipeline([('filter',filtr),('NN',mlp)])
    gs = GridSearchCV(pipe,grid,verbose=10,cv=5)
    
    gs.fit(digitsX,digitsY)
    tmp = pd.DataFrame(gs.cv_results_)
    tmp.to_csv(out+'digits dim red.csv')
#    raise
    #%% data for 3
    # Set this from chart 2 and dump, use clustering script to finish up
    dim = 35
    filtr = ImportanceSelect(rfc,dim)
    
    biodegX2 = filtr.fit_transform(biodegX,biodegY)
    biodeg2 = pd.DataFrame(np.hstack((biodegX2,np.atleast_2d(biodegY).T)))
    cols = list(range(biodeg2.shape[1]))
    cols[-1] = 'Class'
    biodeg2.columns = cols
    biodeg2.to_hdf(out+'datasets.hdf','biodeg',complib='blosc',complevel=9)
    
    dim = 40
    filtr = ImportanceSelect(rfc,dim)
    digitsX2 = filtr.fit_transform(digitsX,digitsY)
    digits2 = pd.DataFrame(np.hstack((digitsX2,np.atleast_2d(digitsY).T)))
    cols = list(range(digits2.shape[1]))
    cols[-1] = 'Class'
    digits2.columns = cols
    digits2.to_hdf(out+'datasets.hdf','digits',complib='blosc',complevel=9)