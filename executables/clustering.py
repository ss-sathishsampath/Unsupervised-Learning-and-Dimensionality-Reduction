# -*- coding: utf-8 -*-
"""
@author : Sathish Sampath(ss.sathishsampath@gmail.com)

"""

#%% Imports
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from time import clock
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans as kmeans
from sklearn.mixture import GaussianMixture as GMM
from collections import defaultdict
from helpers import cluster_acc, myGMM,nn_arch,nn_reg
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import sys

out = './{}/'.format(sys.argv[1])

np.random.seed(0)
digits = pd.read_hdf(out+'datasets.hdf','digits')
digitsX = digits.drop('Class',1).copy().values
digitsY = digits['Class'].copy().values

biodeg = pd.read_hdf(out+'datasets.hdf','biodeg')        
biodegX = biodeg.drop('Class',1).copy().values
biodegY = biodeg['Class'].copy().values


biodegX = StandardScaler().fit_transform(biodegX)
digitsX= StandardScaler().fit_transform(digitsX)

clusters =  [2,5,10,15,20,25,30,35,40]

#%% Data for 1-3
SSE = defaultdict(dict)
ll = defaultdict(dict)
acc = defaultdict(lambda: defaultdict(dict))
adjMI = defaultdict(lambda: defaultdict(dict))
km = kmeans(random_state=5)
gmm = GMM(random_state=5)

st = clock()
for k in clusters:
    km.set_params(n_clusters=k)
    gmm.set_params(n_components=k)
    km.fit(biodegX)
    gmm.fit(biodegX)
    SSE[k]['Biodeg'] = km.score(biodegX)
    ll[k]['Biodeg'] = gmm.score(biodegX)    
    acc[k]['Biodeg']['Kmeans'] = cluster_acc(biodegY,km.predict(biodegX))
    acc[k]['Biodeg']['GMM'] = cluster_acc(biodegY,gmm.predict(biodegX))
    adjMI[k]['Biodeg']['Kmeans'] = ami(biodegY,km.predict(biodegX))
    adjMI[k]['Biodeg']['GMM'] = ami(biodegY,gmm.predict(biodegX))
    
    km.fit(digitsX)
    gmm.fit(digitsX)
    SSE[k]['Digits'] = km.score(digitsX)
    ll[k]['Digits'] = gmm.score(digitsX)
    acc[k]['Digits']['Kmeans'] = cluster_acc(digitsY,km.predict(digitsX))
    acc[k]['Digits']['GMM'] = cluster_acc(digitsY,gmm.predict(digitsX))
    adjMI[k]['Digits']['Kmeans'] = ami(digitsY,km.predict(digitsX))
    adjMI[k]['Digits']['GMM'] = ami(digitsY,gmm.predict(digitsX))
    print(k, clock()-st)
    
    
SSE = (-pd.DataFrame(SSE)).T
SSE.rename(columns = lambda x: x+' SSE (left)',inplace=True)
ll = pd.DataFrame(ll).T
ll.rename(columns = lambda x: x+' log-likelihood',inplace=True)
acc = pd.Panel(acc)
adjMI = pd.Panel(adjMI)


SSE.to_csv(out+'SSE.csv')
ll.to_csv(out+'logliklihood.csv')
acc.ix[:,:,'Digits'].to_csv(out+'Digits acc.csv')
acc.ix[:,:,'Biodeg'].to_csv(out+'Biodeg acc.csv')
adjMI.ix[:,:,'Digits'].to_csv(out+'Digits adjMI.csv')
adjMI.ix[:,:,'Biodeg'].to_csv(out+'Biodeg adjMI.csv')


#%% NN fit data (2,3)

grid ={'km__n_clusters':clusters,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
km = kmeans(random_state=5)
pipe = Pipeline([('km',km),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10)

gs.fit(biodegX,biodegY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'Biodeg cluster Kmeans.csv')


grid ={'gmm__n_components':clusters,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
gmm = myGMM(random_state=5)
pipe = Pipeline([('gmm',gmm),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(biodegX,biodegY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'Biodeg cluster GMM.csv')




grid ={'km__n_clusters':clusters,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
km = kmeans(random_state=5)
pipe = Pipeline([('km',km),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(digitsX,digitsY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'Digits cluster Kmeans.csv')


grid ={'gmm__n_components':clusters,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
gmm = myGMM(random_state=5)
pipe = Pipeline([('gmm',gmm),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(digitsX,digitsY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'Digits cluster GMM.csv')


# %% For chart 4/5
biodegX2D = TSNE(verbose=10,random_state=5).fit_transform(biodegX)
digitsX2D = TSNE(verbose=10,random_state=5).fit_transform(digitsX)

biodeg2D = pd.DataFrame(np.hstack((biodegX2D,np.atleast_2d(biodegY).T)),columns=['x','y','target'])
digits2D = pd.DataFrame(np.hstack((digitsX2D,np.atleast_2d(digitsY).T)),columns=['x','y','target'])

biodeg2D.to_csv(out+'biodeg2D.csv')
digits2D.to_csv(out+'digits2D.csv')


