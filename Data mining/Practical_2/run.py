# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import statistics
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


SensorData=pd.read_csv('.\specs\SensorData_question1.csv')
SensorData['Original Input3']=SensorData['Input3'].copy()
SensorData['Original Input12']=SensorData['Input12'].copy()
#print ("\nZ-score for Input3 : \n", stats.zscore(SensorData['Input3'])) 
#print ("\nZ-score for Input3 : \n", (SensorData['Input3']-SensorData['Input3'].mean())/SensorData['Input3'].std())
SensorData['Input3']=(SensorData['Input3']-SensorData['Input3'].mean())/SensorData['Input3'].std()
SensorData['Input12']=((SensorData['Input12']-SensorData['Input12'].min())/(SensorData['Input12'].max()-SensorData['Input12'].min()))
SensorData['Average Input']=np.mean(SensorData.iloc[:,:13],axis=1)
SensorData.to_csv('./output/question1_out.csv')


DNAData=pd.read_csv('.\specs\DNAData_question2.csv')


#scale the data
scaler = MinMaxScaler()
data_rescaled = scaler.fit_transform(DNAData)


#reduce the number of attributes
pca = PCA(n_components = 0.95)
pca.fit(data_rescaled)
reduced = pca.transform(data_rescaled)
reduced=pd.DataFrame(reduced)

for i in range(reduced.shape[1]):
    DNAData['pca'+str(i)+'_width']=pd.cut(reduced[:,i],10)
for i in range(reduced.shape[1]):
    DNAData['pca'+str(i)+'_freq']=pd.qcut(reduced[:,i],10)
    DNAData.to_csv('./output/question2_out.csv')





