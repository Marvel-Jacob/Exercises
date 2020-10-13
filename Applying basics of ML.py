import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

x=np.array([95,85,80,70,60])
y=np.array([85,95,70,65,70])
xr=np.reshape(x,(5,1))
yr=np.reshape(y,(5,1))
x_mean=np.mean(xr)
y_mean=np.mean(yr)
num=sum((x-x_mean)*(y-y_mean))
den=sum(np.square(x-x_mean))
den
b1=num/den
b1
bo=y_mean-b1*x_mean #y=m(b1)+bo
bo

y_pre=b1*y+bo
y_pre

er1=sum((y-y_mean)**2)
er2=sum((y-y_pre)**2)

R2=1-(er1/er2)
R2

RMSE=(sum((y-y_pre)**2)/5)**0.5
RMSE


#working on datasets

headbrain=pd.read_csv('C:/Users/Administrator/Desktop/Machine learning/Datasets/headbrain.csv')
student=pd.read_csv('C:/Users/Administrator/Desktop/Machine learning/Datasets/student.csv')
train=pd.read_csv('C:/Users/Administrator/Desktop/Machine learning/Datasets/train.csv')

headbrain.head(1)
student.head(2)
train.head(3)


#on headbrain dataset without using the package
x=headbrain['Head Size(cm^3)'].values.reshape(237,1)
y=headbrain['Brain Weight(grams)'].values.reshape(237,1)
x_mean=x.mean()
y_mean=y.mean()
num=sum((x-x_mean)*(y-y_mean))
num
den=sum((x-x_mean)**2)
den
b1=num/den
b1
bo=y_mean-b1*x_mean #y=m(b1)+bo
bo

y_pre=b1*x+bo
y_pre

er1=sum((y-y_mean)**2)
er2=sum((y-y_pre)**2)

R2=1-(er1/er2)
R2

RMSE=(sum((y-y_pre)**2)/headbrain.shape[0])**0.5
RMSE

####################################################################
#using the skitlearn package(linear regression)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#reg=LinearRegression()
#reg=reg.fit(X,Y)
#Y_pred=reg.preD(X)

#rmse=np.sqrt(mean_squared_error(Y, Y_pred))
#r2=reg.score(X,Y)
#r2
####################################################################


#on headbrain dataset using logistic regression package
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error

X=headbrain[['Head Size(cm^3)','Brain Weight(grams)']].values.reshape(237,2)
Y=headbrain['Gender'].values.reshape(237,1)

reg=LogisticRegression()
reg=reg.fit(X,Y)
print('Coeffients: ',reg.coef_)
print('Intercept: ',reg.intercept_)

Y_pred=reg.predict(X)
Y_pred

rmse=np.sqrt(mean_squared_error(Y,Y_pred))
rmse

r2=reg.score(X,Y)
r2

#on train dataset 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

tr=pd.read_csv("C:/Users/Administrator/Desktop/Machine learning/Datasets/train.csv")
tr.head(1)

print(tr['Loan_Status'].value_counts())
tr=tr.dropna()
print(tr['Loan_Status'].value_counts())

tr.drop(['Loan_ID'],axis=1,inplace=True)
tr.columns
tr.head()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
tr.iloc[:,0]=le.fit_transform(tr['Gender'])
tr.iloc[:,1]=le.fit_transform(tr['Married'])
tr.iloc[:,2]=le.fit_transform(tr['Dependents'])
tr.iloc[:,3]=le.fit_transform(tr['Education'])
tr.iloc[:,4]=le.fit_transform(tr['Self_Employed'])
tr.iloc[:,10]=le.fit_transform(tr['Property_Area'])
tr.iloc[:,11]=le.fit_transform(tr['Loan_Status'])

out=tr['Loan_Status']
x=tr.drop(['Loan_Status'],axis=1)
tr

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

Xtrain,xval,Ytrain,yval=train_test_split(x,out,test_size=0.2,random_state=60)

lreg=LogisticRegression()
lreg.fit(Xtrain,Ytrain)
out_pred=lreg.predict(xval)

acc=accuracy_score(yval,out_pred)
acc



from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
model=LogisticRegression()
kfold=KFold(n_splits=5,shuffle=True)
result=cross_val_score(model,Xtrain,Ytrain,cv=kfold,scoring='accuracy')
result

np.mean(result)
np.var(result)
np.std(result)

from sklearn.metrics import confusion_matrix
import seaborn as sb 
cm=confusion_matrix(yval,out_pred)
print('The confusion matrix is:')
print(cm)
plt.figure()
sb.heatmap(cm,annot=True,fmt='d')

plt.subplots()
sb.heatmap(x.corr(),annot=True,linewidth=.5,fmt='.1f')
plt.yticks(rotation=0)

from sklearn.metrics import classification_report
print('The full classification report is as follows:')
print(classification_report(yval,out_pred))

######on cancer dataset
cancer=pd.read_csv("C:/Users/Administrator/Desktop/Machine learning/Datasets/breast_cancer.csv")
cancer.columns

cancer.drop(['id'],inplace=True,axis=1)
cancer1=cancer.dropna()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
cancer.iloc[:,0]=le.fit_transform(cancer1['diagnosis'])

#spliting
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

out=cancer1['diagnosis']
x=cancer.drop(['diagnosis'],axis=1)

xtrain, xval, ytrain, yval=train_test_split(x,out,test_size=0.2,random_state=60)

lreg=LogisticRegression()
lreg.fit(xtrain,ytrain)
out_prediction=lreg.predict(xval)

ac=accuracy_score(yval,out_prediction)
ac

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
model=LogisticRegression()
kfold=KFold(n_splits=5,shuffle=True)
result=cross_val_score(model,xtrain,ytrain,cv=kfold,scoring='accuracy')
result

np.mean(result)
np.var(result)
np.std(result)

from sklearn.metrics import confusion_matrix
import seaborn as sb 
cm=confusion_matrix(yval,out_prediction)
print('The confusion matrix is:')
print(cm)
plt.figure()
sb.heatmap(cm,annot=True,fmt='d')

plt.subplots()
sb.heatmap(x.corr(),annot=True,linewidth=.5,fmt='.1f')
plt.yticks(rotation=0)

from sklearn.metrics import classification_report
print('The full classification report is as follows:')
print(classification_report(yval,out_prediction))


from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
digits=load_digits()
digits
d0=digits.data[0]

d0=d0.reshape(8,8)
plt.imshow(d0,cmap=plt.cm.gray)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(digits.data,digits.target,test_size=0.2,random_state=60)

from sklearn.linear_model import LogisticRegression
lreg=LogisticRegression()
lreg.fit(x_train,y_train)












