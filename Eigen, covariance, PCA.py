#we have to normalise everytime we use PCA

import numpy as np
from numpy.linalg import eig 

A=np.array([[1,2],[3,4],[5,6]])
print(A)

M=np.mean(A.T,axis=1)
C=A-M
V=np.cov(C.T)
print(V)

values,vectors=eig(V)
print(values)
print(vectors)

#projecting the data
P=vectors.T.dot(C.T)
print(P.T)

#Performing PCA
from sklearn.decomposition import PCA
A=np.array([[1,2],[3,4],[5,6]])
#creating a PCA instance
pca=PCA(2)
#fit on data
pca.fit(A)
print(pca.components_)
print(pca.explained_variance_)

p=pca.transform(A)
print(p)

import pandas as pd
bc=pd.read_csv('C:/Users/Administrator/Desktop/Machine learning/Datasets/breast_cancer.csv')
bc1=bc.drop(['diagnosis'],axis=1)
pca=PCA(10) # we take the value as 29(no of columns but since the variance is small then we take the variance which has high variance so we use 10 columns)
pca.fit(bc1)
print(pca.components_)#vectors
print(pca.explained_variance_)#values
P=pca.transform(bc1)


from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
le=LabelEncoder()
lreg=LogisticRegression()

bc=pd.read_csv('C:/Users/Administrator/Desktop/Machine learning/Datasets/breast_cancer.csv')
bc.drop(['id'],axis=1,inplace=True)
x=P #independent columns
y=bc['diagnosis']
y=le.fit_transform(y)

xtrain,xval,ytrain,yval=train_test_split(x,y,test_size=0.2,random_state=60)
lreg.fit(xtrain,ytrain)
out_pred=lreg.predict(xval)
ac=accuracy_score(yval,out_pred)




















