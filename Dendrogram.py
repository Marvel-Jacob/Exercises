import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from scipy.cluster.hierarchy import linkage 
from scipy.cluster.hierarchy import dendrogram 

data=pd.read_csv('C:/Users/Administrator/Desktop/Machine learning/Datasets/company-stock-movements-2010-2015-incl.csv',index_col=0)
data1=normalize(data)
merge=linkage(data1,method='complete')

plt.figure(figsize=(10,5))
dendrogram(merge,labels=data.index)

comp=list(data.index)
