# -*- coding: utf-8 -*-

from sklearn.cluster import KMeans
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn import preprocessing

# =============================================================================
# question1
# =============================================================================
data1=pd.read_csv(".\specs\question_1.csv")
X=data1.values
plt.scatter(X[:,0],X[:,1],c='b')
plt.show()
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
y_kmeans = kmeans.predict(X)
data1['cluster']=y_kmeans
data1.to_csv("./output/question_1.csv")
plt.scatter(X[:,0], X[:,1], c=y_kmeans)
plt.savefig('./output/question_1.pdf')
plt.show()


# =============================================================================
# question2
# =============================================================================
data2=pd.read_csv(".\specs\question_2.csv")
data2=data2.drop(['NAME','MANUF','TYPE','RATING'],axis = 1)
X2=data2.values
kmeans2_1 = KMeans(n_clusters=5,n_init=5,max_iter=100, random_state=0).fit(X2)
data2['config1']=kmeans2_1.labels_
print("Kmeans2_1, Sum of squared distances :",kmeans2_1.inertia_)
kmeans2_2 = KMeans(n_clusters=5,n_init=100,max_iter=100, random_state=0).fit(X2)
data2['config2']=kmeans2_2.labels_
print("Kmeans2_2, Sum of squared distances :",kmeans2_2.inertia_)
kmeans2_3 = KMeans(n_clusters=3,n_init=100,max_iter=100, random_state=0).fit(X2)
data2['config3']=kmeans2_3.labels_
print("Kmeans2_3, Sum of squared distances :",kmeans2_3.inertia_)
data2.to_csv("./output/question_2.csv")

# =============================================================================
# question3
# =============================================================================
data3=pd.read_csv(".\specs\question_3.csv")
data3=data3.drop(['ID'],axis = 1)
X3=data3.values
plt.scatter(X3[:,0],X3[:,1],c='r')
plt.show()
kmeans3_1 = KMeans(n_clusters=7,n_init=5,max_iter=100, random_state=0).fit(X3)
data3['kmeans']=kmeans3_1.labels_
plt.scatter(X3[:,0], X3[:,1], c=kmeans3_1.labels_)
plt.savefig('./output/question_3_1.pdf')
plt.show()

#normalize
min_max_scaler = preprocessing.MinMaxScaler()
X3= min_max_scaler.fit_transform(X3)

#def NormalizeData(data):
#    return (data - np.min(data)) / (np.max(data) - np.min(data))
clustering1 = DBSCAN(eps=0.04, min_samples=4).fit(X3)
data3['dbscan1']=clustering1.labels_
plt.scatter(X3[:,0], X3[:,1], c=clustering1.labels_)
plt.savefig('./output/question_3_2.pdf')
plt.show()

clustering2 = DBSCAN(eps=0.08, min_samples=4).fit(X3)
data3['dbscan2']=clustering2.labels_
plt.scatter(X3[:,0], X3[:,1], c=clustering2.labels_)
plt.savefig('./output/question_3_3.pdf')
plt.show()
data3.to_csv("./output/question_3.csv")


