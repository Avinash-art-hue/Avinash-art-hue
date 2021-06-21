# AVINASH PANDEY
MSC(DATASCIENCE)

# PREDICATION BASED ON UNSUPERVISED LEARNING USING IRIS DATASET

## Predict the optimum number of clusters and represent it visually.

#importing all the libraries.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#importing and reaading the dataset
dataset = pd.read_csv("/home/avinash/Downloads/Iris.csv")

#the first five values in the dataset
dataset.head()

#getting all the unique values in SepalLengthCm
dataset["SepalLengthCm"].unique()

##getting all the unique values in SepalWidthCm
dataset["SepalWidthCm"].unique()

#number of rows and columns
dataset.shape

#Number of species in the dataset
dataset.groupby(["Species"]).count()

dataset.describe()

## Visualization

sns.countplot(x='Species',data=dataset)
plt.title('Species',fontsize=20)
plt.show()

dataset.plot(kind ="scatter", 
          x ='SepalLengthCm', 
          y ='PetalLengthCm') 
plt.grid()

sns.scatterplot(x=dataset["SepalLengthCm"], y=dataset["SepalWidthCm"], hue=dataset["Species"])
plt.show()

sns.scatterplot(x=dataset["PetalLengthCm"], y=dataset["PetalWidthCm"], hue=dataset["Species"])
plt.show()

## Using the elbow method to find the optimal number of clusters

#Taking values except for "id" and "Species"
X = dataset.iloc[:, [1,2,3,4]].values
print(X)

#k-means clustering
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 15):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 15), wcss)
plt.title(' Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

## Training the K-Means model on the dataset

kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)
print(y_kmeans)

## Visualising the clusters

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], 
            s = 100, c = 'magenta', label = 'Iris-setosa')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], 
            s = 100, c = 'cyan', label = 'Iris-versicolour')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1],
            s = 100, c = 'orange', label = 'Iris-virginica')

# Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], 
            s = 200, c = 'red', label = 'Centroids')
plt.title('Clusters of Iris Species')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
plt.show()

## Using the dendrogram to find the optimal number of clusters

#Hierarchical Clustering
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Species')
plt.ylabel('Euclidean distances')
plt.show()

## Training the Hierarchical Clustering model on the dataset

#AgglomerativeClustering
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)

## Visualising the clusters

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], 
            s = 100, c = 'red', label = 'Iris-setosa')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], 
            s = 100, c = 'blue', label = 'Iris-versicolour')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1],
            s = 100, c = 'orange', label = 'Iris-virginica')

plt.title('Clusters of Iris Species')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
plt.show()

## Conclusion
# We were able to predict the optimum number of clusters and represent it visually from the given ‘Iris’ dataset.




