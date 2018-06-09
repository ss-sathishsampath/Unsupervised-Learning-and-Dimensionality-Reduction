# -*- coding: utf-8 -*-
"""
@author : Sathish Sampath(ss.sathishsampath@gmail.com)

"""


import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from helpers import   nn_arch,nn_reg
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

out = './BASE/'
np.random.seed(0)
digits = pd.read_hdf('./BASE/datasets.hdf','digits')
digitsX = digits.drop('Class',1).copy().values
digitsY = digits['Class'].copy().values

biodeg = pd.read_hdf('./BASE/datasets.hdf','biodeg')        
biodegX = biodeg.drop('Class',1).copy().values
biodegY = biodeg['Class'].copy().values


biodegX = StandardScaler().fit_transform(biodegX)
digitsX= StandardScaler().fit_transform(digitsX)

#%% benchmarking for chart type 2

grid ={'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
pipe = Pipeline([('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(biodegX,biodegY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'Biodeg NN bmk.csv')


mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
pipe = Pipeline([('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(digitsX,digitsY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'digits NN bmk.csv')
raise