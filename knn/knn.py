# Import libraries

import pandas as pd
from numpy import square, sqrt, sum, random, asarray, meshgrid, c_,arange, shape
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from collections import Counter
from time import process_time, perf_counter
import  matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

def getData(n):
    df =  pd.read_pickle('df_clean')
    # dataset
    X = df.iloc[:,1:3].as_matrix()
    y = df.iloc[:,4].as_matrix()
    # Only for small visualization
    
    indices = random.randint(0,X.shape[0],n)
    X = X[indices]
    y = y[indices]
    X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=0)
    return (X_train, X_test, y_train, y_test)

def kNNeighbor(X_train, y_train, X_test, y_pred, k):
	# raise an error if k is larger than n
	if k > len(X_train):
		raise ValueError

	# prediction of each sample
	for i in range(len(X_test)):
		y_pred.append(makePred(X_train, y_train, X_test[i, :], k))



def makePred(X_train, y_train, X_test, k):
	# create list for distances and targets
	distances = []
	targets = []

	for i in range(len(X_train)):
		# Computing the quadratic distance
		distance = sqrt(sum(square(X_test - X_train[i, :])))
		# append to the list of distances
		distances.append([distance, i])

	# sort the list
	distances = sorted(distances)

	# Get the k neighbours of the target
	for i in range(k):
		index = distances[i][1]
		targets.append(y_train[index])

	# return most common target
	return Counter(targets).most_common(1)[0][0]

def plot_scatter_description(X,y):
    plt.figure()
    plt.title('Description of towns in French presidential elections')
    cmap_bold = ListedColormap(['#FFFF00', '#FF00AA', '#000000', '#00AAFF'])
    color_array = ['#FFFF00', '#00AAFF', '#000000', '#FF00AA','r','b','g','k']
    plt.scatter(X[:, 0], X[:, 1], c=y, marker= 'o', s=50, cmap=cmap_bold)
    plt.xlabel('abstention/subscribers %')
    plt.ylabel('whiteballot/subscribers %')
    h = []
    for c in range(0, 2):
        h.append(mpatches.Patch(color=color_array[c], label=["MACRON","LE PEN"][c]))
    plt.legend(handles=h)
    plt.show()
    
def plot_scatter_knn(X,y,k):
    
    indices = random.randint(0,X.shape[0],k)
    Xsmall = X[indices]
    ysmall = y[indices]
    # from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(Xsmall, y,random_state=0)
    plot_knn(X_train, y_train, m_neighbours, 'uniform', X_test, y_test)
    return Xsmall, ysmall

def plot_knn(X, y, n_neighbors, weights, X_test, y_test):
    X_mat = X
    y_mat = y

    # Create color maps
    cmap_light = ListedColormap(['#FFFFAA', '#AAFFAA', '#AAAAFF','#EFEFEF'])
    cmap_bold  = ListedColormap(['#FFFF00', '#00FF00', '#0000FF','#000000'])

    clf = KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(X_mat, y_mat)

    # Plot the decision boundary by assigning a color in the color map
    # to each mesh point.
    
    mesh_step_size = .01  # step size in the mesh
    plot_symbol_size = 50
    
    x_min, x_max = X_mat[:, 0].min() - 1, X_mat[:, 0].max() + 1
    y_min, y_max = X_mat[:, 1].min() - 1, X_mat[:, 1].max() + 1
    xx, yy = meshgrid(arange(x_min, x_max, mesh_step_size),
                         arange(y_min, y_max, mesh_step_size))
    Z = clf.predict(c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot training points
    plt.scatter(X_mat[:, 0], X_mat[:, 1], s=plot_symbol_size, c=y, cmap=cmap_bold, edgecolor = 'black')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    title = "Neighbors = {}".format(n_neighbors)
    if (X_test is not None):
        train_score = clf.score(X_mat, y_mat)
        test_score  = clf.score(X_test, y_test)
        title = title + "\nTrain score = {:.2f}, Test score = {:.2f}".format(train_score, test_score)

    patch0 = mpatches.Patch(color='#FFFF00', label='LE PEN')
    patch1 = mpatches.Patch(color='#000000', label='MACRON')
    plt.legend(handles=[patch0, patch1])

    plt.xlabel('Feature 0')
    plt.ylabel('Feature 1')
    plt.title(title)

    plt.show()



# List of predicitons
y_pred = []
y_trainPred = []
try:
    # number of neighbours
    m_neighbours = 7 
    # n row (randomly) from data
    n = 2000
    # k row (randomly) for scatter visualization 
    k = 100
    # Get data
    X_train, X_test, y_train, y_test = getData(n)
    print("==================================================")
    print(">>> Data set details <<<" )
    print("==================================================")
    print("number of samples : ", n)
    print("number of neighbours : ", m_neighbours)
    print("shape of training set", shape(X_train))
    print("shape of test set", shape(X_test))
    print("==================================================")
    print(">>> Implemented KNN simulation <<<" )
    print("==================================================")
    t_per_start = perf_counter()
    t_pro_start = process_time()
        
    # KNN
    kNNeighbor(X_train, y_train, X_test, y_pred, m_neighbours)
    # make prediction for test set
    y_pred = asarray(y_pred)
    
    # KNN
    kNNeighbor(X_train, y_train, X_train, y_trainPred, m_neighbours)
    # make prediction for train set
    y_trainPred = asarray(y_train)
    
    t_per_stop = perf_counter()
    t_pro_stop = process_time()
    print("--------------------------------------------------")
    print("Elapsed time: %.1f [min]" % ((t_per_stop-t_per_start)/60))
    print("CPU process time: %.1f [min]" % ((t_pro_stop-t_pro_start)/60))
    print("--------------------------------------------------") 
    
    
    print("==================================================")
    print(">>> SCKIT LEARN KNN simulation <<<" )
    print("==================================================")
    t_per_start1 = perf_counter()
    t_pro_start1 = process_time()
    # instantiate learning model (k)
    knn = KNeighborsClassifier(n_neighbors=m_neighbours)

    # fitting the model
    knn.fit(X_train, y_train)

    # predict the response
    pred = knn.predict(X_test)

    # predict the response
    preTrain = knn.predict(X_train)
    t_per_stop1 = perf_counter()
    t_pro_stop1 = process_time()
    print("--------------------------------------------------")
    print("Elapsed time: %.1f [min]" % ((t_per_stop1-t_per_start1)/60))
    print("CPU process time: %.1f [min]" % ((t_pro_stop1-t_pro_start1)/60))
    print("--------------------------------------------------") 

    	# evaluating accuracy
    print("--------------------------------------------------")
    print("Impelmented KNN")
    print("Accuracy of train set : %d%% " % (accuracy_score(y_train, y_trainPred) * 100))
    print("Accuracy of test set : %d%%" % (accuracy_score(y_test, y_pred) * 100))
        
    # evaluate accuracy
    print("--------------------------------------------------")
    print("sckit learn KNN")
    print("Accuracy of train set : %d%% " % (accuracy_score(y_train, preTrain) * 100))
    print("Accuracy of test set : %d%%" % (accuracy_score(y_test, pred) * 100))    
    
    # visualize a small portion of the knn scatter
    Xsmall, ysmall = plot_scatter_knn(X_train,y_train,k)
    
    # visualize a description of data
    plot_scatter_description(Xsmall, ysmall)

except ValueError:
    print('Not possible to have more neighbors than samples !')