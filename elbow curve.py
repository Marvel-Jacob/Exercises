#######  ELBOW CURVE  #############    
#when the number of clusters are more the error start reducing it will reduce till
#the time the k value reaches the value of number of datapoints. at some point it 
#will have a sharp point which we consider an elbow curve.



from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

x1=np.array([3,1,1,2,1,6,6,6,5,6,7,8,9,8,9,9,8])
x2=np.array([5,4,5,6,5,8,6,7,6,7,1,2,1,2,3,2,3])

plt.plot()
plt.xlim([0,10])
plt.ylim([0,10])
plt.title('dataset')
plt.scatter(x1,x2)
plt.show()

x=np.array(list(zip(x1,x2)))

distortions=[]
K=range(1,10)
for k in K:
    kmeanModel=KMeans(n_clusters=k)
    kmeanModel.fit(x)
    distortions.append(sum(np.min(cdist(x,
                                        kmeanModel.cluster_centers_,
                                        'euclidean'),axis=1))/x.shape[0])
    
plt.plot(K,distortions,'bx-')
plt.xlabel('k')
plt.ylabel('Distortions')
plt.scatter(x1,x2)
plt.title('The elbow method showing optimal K')
plt.show()

#### using wholesale customer data #####

data=pd.read_csv('C:/Users/Administrator/Desktop/Machine learning/Datasets/Wholesale customers data.csv')
data.drop(['Channel','Region'],axis=1,inplace=True)

################applying PCA#####################
from sklearn.decomposition import PCA
pca=PCA(3)
pca.fit(data)
print(pca.components_)
print(pca.explained_variance_)
p=pca.transform(data)

from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
from sklearn import cluster
from sklearn.metrics import silhouette_score

n_clusters=[10,9,8,7,6,5,4,3,2]
for n in n_clusters:
    clust=cluster.KMeans(n_clusters=n).fit(p)
    pred=clust.predict(p)
    centers=clust.cluster_centers_
    score=silhouette_score(p,pred)
    print("The silhouette_score for {} clusters is {} ".format(n,score))




#####################################


#find the optimum k
distortions=[] #errors
K=range(1,10)
for k in K:
    model=KMeans(n_clusters=k)
    model.fit(data)
    distortions.append(sum(np.min(cdist(data,model.cluster_centers_,
                                        'euclidean'),axis=1))/data.shape[0])
    
plt.plot(K,distortions,'bx-')
plt.xlabel('k')
plt.ylabel('Distortions')
plt.title('The elbow method showing optimal K')
plt.show()

#since we cant tell the proper k value from the plot
n_clusters=[10,9,8,7,6,5,4,3,2]
from sklearn import cluster
from sklearn.metrics import silhouette_score

for n in n_clusters:
    clust=cluster.KMeans(n_clusters=n).fit(data)
    pred=clust.predict(data)
    centers=clust.cluster_centers_
    score=silhouette_score(data,pred)
    print("The silhouette_score for {} clusters is {} ".format(n,score))
#silhouette score is the distance bwteen the clusters. The silhouette score
# should be max but the datapoints need to be close distance within a cluster
 
#################################    
#have to install kelbowvisualizer
model=cluster.KMeans()
from yellowbrick.cluster import KElbowVisualizer
kelb_graph=KElbowVisualizer(model,k=(1,8))
kelb_graph.fit(data)
kelb_graph.poof 
##################################

clust_range=range(1,10)
clust_err=[]
for num_clust in clust_range:
    cluster=cluster.KMeans(num_clust)
    cluster.fit(data)
    clust_err.append(cluster.inertia_)

cluster_df=pd.DataFrame({"Num_cluster":clust_range,"Cluster_err":clust_err})
cluster_df[0:10]




    
    