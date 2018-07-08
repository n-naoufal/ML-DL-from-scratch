# Import libraries

import pandas as pd
from numpy import array, arange, where, linalg, average, sum, random
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from time import process_time, perf_counter
import  matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches
from matplotlib.font_manager import FontProperties

def getData():
    df =  pd.read_pickle('df_clean')
    # dataset
    X = df.iloc[:,1:4].as_matrix()
    listTowns = df.town
    # get the label axis list
    listAxis = list(df.iloc[:,1:4].columns)
    return (X, listTowns, listAxis)

def fit(X, k, max_iter, tol):
    centroids = {}
    #initialize the centroids with the first 'k' elements
    for i in range(k):
        centroids[i] = X[i]
    # iterate
    for i in range(max_iter):
        classes = {}
        for i in range(k):
            classes[i] = []
        #Choose the nearest centroid
        for col in X:
            distances = [linalg.norm(col - centroids[point]) for point in centroids]
            min_distance = distances.index(min(distances))
            classes[min_distance].append(col)
        # store previous
        old = dict(centroids)
			#Update the centroids
        for clas in classes:
            centroids[clas] = average(classes[clas], axis = 0)
        for point in centroids:
            old_centroid = old[point]
            curr = centroids[point]
            # break if positions don't change anymore
            if sum((curr - old_centroid)/old_centroid * 100.0) <= tol:
                break
    return centroids,classes

def predict(X, centroids):
    distances = [linalg.norm(X - centroids[point]) for point in centroids]
    clas = distances.index(min(distances))
    return clas

def plot_user_scatter(X, y, class_centers, class_labels, i, j, listAxis):
    
    num_labels = len(class_labels)
    x_min, x_max = X[:, i].min() - 1, X[:, i].max() + 1
    y_min, y_max = X[:, j].min() - 1, X[:, j].max() + 1

    color_array = ['#FFFF00', '#00AAFF', '#000000', '#FF00AA','R','B','G','K',
                  '#F5F5DC','#BEBEBE','#F0E68C','#D2691E','#331E28','#FF99CC','#99FFCC',
            '#CCFFE6','#FFCCFF','#CCCCFF','#FFFFCC','#979785','#488997','#EDEEE3',
            '#5CB0C2','#5199A8','#07A3B2','#BA3951','#4B0D2B','#E9B704','#8E3843',
            '#134913','#C50D63','#CD7EA0','#596B53','#9A616E','#82486B','#95818C',
            '#07A3B2','#C50D63','#E47446','#055F65','#CBAA5C','#076E4E']
    
    cmap_bold = ListedColormap(color_array)
    bnorm = BoundaryNorm(arange(0, num_labels + 1, 1), ncolors=num_labels)
    plt.figure(figsize=(8, 6))
    plt.scatter(class_centers[:,i],class_centers[:,j], cmap=cmap_bold, s=200, alpha=0.5, marker="x")
    plt.scatter(X[:, i], X[:, j], s=65, c=y, cmap=cmap_bold, norm = bnorm, alpha = 0.40, edgecolor='black', lw = 1)
    plt.xlabel('{}'.format(listAxis[i]))
    plt.ylabel('{}'.format(listAxis[j]))
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    h = []
    for c in range(0, num_labels):
        h.append(mpatches.Patch(color=color_array[c], label=class_labels[c]))
    fontP = FontProperties()
    fontP.set_size('small')
    plt.legend(handles=h,prop=fontP)
    plt.title("clustering of French towns in regards to vote behaviours")
    plt.show()


# number of clusters
nclusters = 5 
# tolerance
tol =1e-4;
# max iterations
max_iter = 100
# Get data
X, listTowns, listAxis = getData()
# get a portion of the data to simplify visualization
perm = random.permutation(X.shape[0])
X = X[perm[:500]]
# normalize data 
X_normalized = StandardScaler().fit(X).transform(X)  
# get labels
labels = array(list(listTowns))
print("==================================================")
print(">>> Data set details <<<" )
print("==================================================")
print("number of samples : ", X.shape[0])
print("number of clusters : ", nclusters)
print("==================================================")
print(">>> Implemented Kmeans CLUSTERING simulation <<<" )
print("==================================================")
t_per_start = perf_counter()
t_pro_start = process_time()
    
centroids, classes = fit(X, nclusters, max_iter, tol)
#cla = predict(X, centroids)
    
t_per_stop = perf_counter()
t_pro_stop = process_time()
print("--------------------------------------------------")
print("Elapsed time: %.1f [min]" % ((t_per_stop-t_per_start)/60))
print("CPU process time: %.1f [min]" % ((t_pro_stop-t_pro_start)/60))
print("--------------------------------------------------") 


# Plotting starts here
colors = 10*["r", "g", "c", "b", "k"]
plt.figure()
for point in centroids:
    plt.scatter(centroids[point][0], centroids[point][1], s = 130, marker = "x")
for cla in classes:
		color = colors[cla]
		for cols in classes[cla]:
			plt.scatter(cols[0], cols[1], color = color,s = 30)
plt.show()

print("==================================================")
print(">>> SCKIT LEARN Kmeans CLUSTERING simulation <<<" )
print("==================================================")
t_per_start1 = perf_counter()
t_pro_start1 = process_time()
# applying the model
kmeans = KMeans(n_clusters = nclusters, max_iter = max_iter, tol = tol)
kmeans.fit(X)
pred_classes = kmeans.predict(X)


# Print different clusters
for cluster in range(nclusters):
    print('')
    print('cluster: ', cluster+1)
    print(labels[where(pred_classes == cluster)])

# get the label axis list

t_per_stop1 = perf_counter()
t_pro_stop1 = process_time()
print("--------------------------------------------------")
print("Elapsed time: %.1f [min]" % ((t_per_stop1-t_per_start1)/60))
print("CPU process time: %.1f [min]" % ((t_pro_stop1-t_pro_start1)/60))
print("--------------------------------------------------")

# visualize the kmeans clustering scatter
plot_user_scatter(X, kmeans.labels_,kmeans.cluster_centers_, ['Cluster {}'.format(i+1) for i in range(nclusters) ], 0,  1, listAxis)
plot_user_scatter(X, kmeans.labels_,kmeans.cluster_centers_, ['Cluster {}'.format(i+1) for i in range(nclusters) ], 0,  2, listAxis)
plot_user_scatter(X, kmeans.labels_,kmeans.cluster_centers_, ['Cluster {}'.format(i+1) for i in range(nclusters) ], 1,  2, listAxis)

