# k-Nearest-Neighbors (kNN) implementation from scratch

The kNN algorithm is a versatile Non-parametric classifier that is often used as a benchmark for more complex classifiers such as Artificial Neural Networks (ANN) and Support Vector Machines (SVM). It is a type of instance-based learning, or lazy learning, where the function is only approximated locally and all computation is deferred until classification. The kNN algorithm is among the simplest of all machine learning algorithms.
Despite its simplicity, kNN can outperform more powerful classifiers and is used in a variety of applications such as economic forecasting, data compression and genetics. It has been used in statistical estimation and pattern recognition already in the beginning of 1970’s as a non-parametric technique. 

## The dataset 
National renewal is what both the rival French presidential candidates are promising, but they offer very different paths to get there.

Liberal centrist Emmanuel Macron - winner of the first round - and nationalist Marine Le Pen are already revolutionising French politics. Mr Macron leads a new movement called En Marche (On the Move), while Ms Le Pen is backed by the National Front (FN). They disagree on many issues, especially Europe and immigration.

The decisive second round was decisive and polls showed Mr Macron with a firm lead. But what are the main patterns of the towns that support Macron and Le Pen visions?

The data set was retrieved from https://www.data.gouv.fr/fr/posts/les-donnees-des-elections/ which is the open public data platform "data.gouv.fr" that hosts the data sets and records their reuses.

The data set is cleaned in Jupyter notebook checking_cleaning_data.ipynb and is as follows: 
![alt text](figures/dataset.PNG?raw=true "")

with three features: 
* the ratio abstention/subscribers  
* the ratio whiteballot/subscribers 
* the ratio invalid/voters

A description of a small portion of the dataset is presented in the figure below:
![alt text](figures/descrip_data.png?raw=true "")


## The implemented KNN

We can implement a KNN model by following the below steps:

```
* Get the data
* Initialise the value of k
* Iterate from 1 to total number of training data samples
-----> Calculate the distance between test data and each row of training data.
-----> Sort the calculated distances in ascending order based on distance values
-----> Get top k rows from the sorted array
-----> Get the most frequent class of these rows
* Return the prediction
```

![alt text](figures/knn_plot.png?raw=true "")

## Comparison to Sckit Learn kNN

scikit-learn implements kNeighborsClassifier based on the k nearest neighbors of each query point, where k is an integer value specified by the user. It is the more commonly used and the optimal choice of the value k is highly data-dependent: in general a larger k suppresses the effects of noise, but makes the classification boundaries less distinct. The basic nearest neighbors classification uses uniform weights: that is, the value assigned to a query point is computed from a simple majority vote of the nearest neighbors. Under some circumstances, it is better to weight the neighbors such that nearer neighbors contribute more to the fit. This can be accomplished through the weights keyword. The default value, weights = 'uniform', assigns uniform weights to each neighbor. weights = 'distance' assigns weights proportional to the inverse of the distance from the query point. Alternatively, a user-defined function of the distance can be supplied which is used to compute the weights.

Here, we present a comparison between the implemented kNN and the Sckit Learn one. 
![alt text](figures/output.PNG?raw=true "")


## Final remarks 

One of the obvious drawbacks of the KNN algorithm is the computationally expensive testing phase which is impractical in industry settings. Also, the accuracy of KNN can be severely degraded with high-dimension data because there is little difference between the nearest and farthest neighbor.

