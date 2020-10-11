# -*- coding: utf-8 -*-
"""
Predicitive_Analytics.py
"""

# Importing the required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy

# Importing the data
data = np.genfromtxt('data.csv', delimiter = ',', skip_header = 1)

def Accuracy(y_true,y_pred):
    """
    :type y_true: numpy.ndarray
    :type y_pred: numpy.ndarray
    :rtype: float
    """
    
    count = 0
    
    for i in range(len(y_true)):
        if y_true[i]==y_pred[i]:
            count+=1
            
    return count/len(y_true)

def Recall(y_true,y_pred):

    from collections import defaultdict 
    rec_class = np.array([])
    dict_truecnt = defaultdict(int)
    
    for i, y in enumerate(y_true):
        if y_true[i]==y_pred[i]:
            dict_truecnt[y] += 1
            
    cnt_class = np.bincount(y_true)

    for key in dict_truecnt:
        if cnt_class[key]!=0:
            a = dict_truecnt[key]/cnt_class[key]
            rec_class = np.append(rec_class, a)
    return np.sum(rec_class)/len(np.unique(y_true))

def Precision(y_true,y_pred):
    """
    :type y_true: numpy.ndarray
    :type y_pred: numpy.ndarray
    :rtype: float
    """

    from collections import defaultdict 
    pres_class = np.array([])
    dict_truecnt = defaultdict(int)
    
    for i, y in enumerate(y_true):
        if y_true[i]==y_pred[i]:
            dict_truecnt[y] += 1
            
    cnt_class = np.bincount(y_pred)

    
    for key in dict_truecnt:
        if cnt_class[key]!=0:
            a = dict_truecnt[key]/cnt_class[key]
            pres_class = np.append(pres_class, a)
    return np.sum(pres_class)/len(np.unique(y_true))


def WCSS(Clusters):
    """
    :Clusters List[numpy.ndarray]
    :rtype: float
    """
    wcss_sum = 0

    for cluster in Clusters:
        cluster_mean = np.mean(cluster, 0)
        wcss_sum += np.sum(np.square(np.linalg.norm(cluster - cluster_mean, axis=1)))
    return wcss_sum

def ConfusionMatrix(y_true,y_pred):
    
    """
    :type y_true: numpy.ndarray
    :type y_pred: numpy.ndarray
    :rtype: float
    """  
    y_true = pd.Series(y_true, name='Actual')
    y_pred = pd.Series(y_pred, name='Predicted')
    confusion_matrix = pd.crosstab(y_true, y_pred)
    
    fig, ax = plt.subplots()
    ax.imshow(confusion_matrix, cmap= 'cool')
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            ax.text(j, i, "{}".format(confusion_matrix.iloc[i,j]), ha="center", va="center",color="r")
    
    return confusion_matrix

def KNN(X_train,X_test,Y_train,k):
    
    if k==0:
        return 'not a valid k'
    
    # Standardization of data
    X_train = X_train - np.mean(X_train, axis = 0)
    X_train = X_train / np.std(X_train, axis = 0)
    X_test = X_test - np.mean(X_test, axis = 0)
    X_test = X_test / np.std(X_test, axis = 0)
    
    Y_test =np.array([],int)   
    
    for x_test in X_test:
        
        x_test = np.tile(x_test,(X_train.shape[0],1))
        distance = np.linalg.norm(X_train - x_test,axis = 1)
        j = np.argsort(distance)[:k,]
        labels_y = Y_train[j].astype(int)
        a = np.bincount(labels_y).argmax()
        Y_test = np.append(Y_test,a)
        
        
    return Y_test


def RandomForest(X_train,Y_train,X_test):
    """
    :type X_train: numpy.ndarray
    :type X_test: numpy.ndarray
    :type Y_train: numpy.ndarray
    
    :rtype: numpy.ndarray
    """
    
    def bootstrap_sampling(X_train, Y_train):


        idx = np.random.randint(len(Y_train), size = int(0.7*len(Y_train))) 
        X_train = X_train[idx]
        Y_train = Y_train[idx].astype(int)
        
    
        return X_train, Y_train
    

    #---------------------------------------------------------------------------------------------------------------
    
    def calculate_gini(Y):
    
        cls, count = np.unique(Y, return_counts = True)
        j = len(Y)
        gini = 1

        for cnt in count:
            gini -= (cnt/j)**2 

        return gini

    #----------------------------------------------------------------------------------------------------------
    
    def best_split(X_train, Y_train):
        best_f, best_thre = None, None
        x = len(X_train)

        min_gini = calculate_gini(Y_train)

        for i, f in enumerate(X_train.T):

            a = np.argsort(X_train[:,i])
            new_y = Y_train[a]

            for j in range(1,x):

                Y_lef = new_y[0:j]
                Y_righ = new_y[j:]
                gini_left = calculate_gini(Y_lef)
                gini_right = calculate_gini(Y_righ)
                wtd_gini_childs = ((j*gini_left)+(x-j)*gini_right)/x

                if f[j-1] == f[j]:
                    continue

                if wtd_gini_childs < min_gini:
                    min_gini = wtd_gini_childs
                    best_thre = (f[j]+f[j-1])/2
                    best_f = i 

    
        return best_f, best_thre

#______________________________________

    def grow_tree(X,Y,depth = 0):
        max_depth = 5

        if depth >= max_depth:
            return  {'class' : np.bincount(Y).argmax()}

        elif len(Y) == 0:
            return None

        elif len(np.unique(Y)) == 1:
            return {'class' : Y[0]}

        else:
            idx_f, thre = best_split(X, Y)

            if idx_f is not None:
                idx_left = X[:,idx_f] < thre
                X_left, Y_left = X[idx_left], Y[idx_left]
                X_right, Y_right = X[~idx_left], Y[~idx_left]

                if len(Y_left)!=0 and len(Y_right) != 0:
                    node_inf = {'idx-col': idx_f, 'threshold': thre, 'class' : np.bincount(Y).argmax() }
                    node_inf['Left'] = grow_tree(X_left, Y_left, depth+1)
                    node_inf['right'] = grow_tree(X_right, Y_right, depth+1)

                else:
                    return {'class' : np.bincount(Y).argmax()}
        return node_inf  
 #------------------------------------------------------------------------------------------------------------------- 
    
    
    def predict_tree(X_test,tree):

        X_test = X_test - np.mean(X_test, axis = 0)
        X_test = X_test / np.std(X_test, axis = 0)

        predictions = np.array([])
        tr = tree

        for row in X_test:
            tree = tr

            while  tree!=None and tree.get('threshold'):

                if row[tree['idx-col']] < tree['threshold']:
                    tree = tree['Left']

                else:
                    tree = tree['right']

            predictions = np.append(predictions, tree.get('class'))

        return predictions
    
    #------------------------------------------------------------------------------------------------------------
    


    
    trees =np.array([])
    for i in range(5):   # taking 9 bootstrap samples

        X,Y = bootstrap_sampling(X_train,Y_train)

        X = X - np.mean(X, axis = 0)
        X = X / np.std(X, axis = 0)

        tree  = grow_tree(X, Y, depth = 0) # growing tree for each sample
        trees = np.append(trees, tree)

    preds = np.array([predict_tree(X_test,tree) for tree in trees]).astype(int) # predicting for each tree
    final_pred = np.array([np.bincount(row).argmax() for row in preds.T]) # final prediction aggregating all the models
    
    
    return final_pred
    
def PCA(X_train,N):
    """
    :type X_train: numpy.ndarray
    :type N: int
    :rtype: numpy.ndarray
    """
    if N > X_train.shape[1]:
        print("No.of dimensions must be smaller than the no.of feautures")
        return None
    # Standardization of data
    X_std = X_train - np.mean(X_train, axis = 0)
    X_std = X_std / np.std(X_std, axis = 0)
    
    X = X_std
    # Covariance Matrix
    covar_mat = np.matmul(X.T, X) * (1/X.shape[0])
    
    # Calculating the eigen values and vectors corresponding to N
    eig_vals, eig_vectors = np.linalg.eigh(covar_mat)
    eig_vectors = eig_vectors[:, eig_vals.shape[0] : eig_vals.shape[0] - N -1 : -1]
    eig_vectors = eig_vectors.T
    
    # The final data
    data = np.matmul(eig_vectors, X.T)
    
    return data
    
def Kmeans(X_train,N):
    """
    :type X_train: numpy.ndarray
    :type N: int
    :rtype: List[numpy.ndarray]
    """
    # Standardization of data
    X_std = X_train - np.mean(X_train, axis = 0)
    X_std = X_std / np.std(X_std, axis = 0)
    n_samples = X_std.shape[0]
    n_features = X_std.shape[1]
    
    # Initializing the random N number of centers
    np.random.seed(4)
    centers = np.random.randn(N, n_features)
    
    centers_old = np.zeros(centers.shape)
    centers_updated = deepcopy(centers)

    cluster_indices = np.zeros(n_samples)
    clusters = []
    distances = np.zeros((n_samples,N))

    change = np.linalg.norm(centers_updated - centers_old)

    while change != 0:
        
        # Distance from the points to the N centers
        for cls in range(N):
            distances[:,cls] = np.linalg.norm(X_std - centers[cls], axis=1)
            
        # Assigning each point to its nearest center
        cluster_indices = np.argmin(distances, axis = 1)

        centers_old = deepcopy(centers_updated)
        # Finding out the mean for the updated centers
        for cls in range(N): 
            centers_updated[cls] = np.mean(X_std[cluster_indices == cls], axis=0)
            
        change = np.linalg.norm(centers_updated - centers_old)
    
    for i in range(N):
        sample = X_std[cluster_indices == i]
        clusters.append(sample)
    
    # Returning the final clusters
    clusters = np.array(clusters)
    return clusters

def SklearnSupervisedLearning(X_train,Y_train,X_test):
    """
    :type X_train: numpy.ndarray
    :type X_test: numpy.ndarray
    :type Y_train: numpy.ndarray
    
    :rtype: List[numpy.ndarray] 
    """
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    scaler.fit(X_test)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    result = []

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    scaler.fit(X_test)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    # SVM
    from sklearn.svm import SVC
    svc = SVC()
    svc.fit(X_train,Y_train)
    predictions_svc = svc.predict(X_test)
    result.append(predictions_svc)
    #print('The accuracy for SVM is: ', Accuracy(y_true,predictions_svc))
    #ConfusionMatrix(y_true,predictions_svc)
    
    # Logistic Regression
    from sklearn.linear_model import LogisticRegression
    logistic = LogisticRegression()
    logistic.fit(X_train,Y_train)
    predictions_lr = logistic.predict(X_test)
    result.append(predictions_lr)
    #print('The accuracy for Logistic Regression is: ', Accuracy(y_true,predictions_lr))
    #ConfusionMatrix(y_true,predictions_lr)
    
    # Decision Tree
    from sklearn.tree import DecisionTreeClassifier
    decision = DecisionTreeClassifier()
    decision.fit(X_train,Y_train)
    predictions_dt = decision.predict(X_test)
    result.append(predictions_dt)
    #print('The accuracy for Decision Tree is: ', Accuracy(y_true,predictions_lr))    
    #ConfusionMatrix(y_true,predictions_lr)
    
    
    # KNN
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train,Y_train)
    predictions_knn = knn.predict(X_test)
    result.append(predictions_svc)
    #print('The accuracy for KNN is: ', Accuracy(y_true,predictions_knn))
    #ConfusionMatrix(y_true,predictions_knn)

    result = np.array(result)
    return result

def SklearnVotingClassifier(X_train,Y_train,X_test):
    
    """
    :type X_train: numpy.ndarray
    :type X_test: numpy.ndarray
    :type Y_train: numpy.ndarray
    
    :rtype: List[numpy.ndarray] 
    """
    result = []

    from sklearn.ensemble import VotingClassifier
    estimators=[('knn', knn), ('logistic', logistic), ('decision', decision)]
    ensemble = VotingClassifier(estimators, voting='hard')
    ensemble.fit(X_train, Y_train)
    predictions = ensemble.predict(X_test)
    result.append(predictions)
    #print('The accuracy for Voting Classifier is: ', Accuracy(y_true,predictions))
    #ConfusionMatrix(y_true,predictions)

"""
Create your own custom functions for Matplotlib visualization of hyperparameter search. 
Make sure that plots are labeled and proper legends are used
"""
%matplotlib inline
def confusion_matrix_plots(X_train,Y_train,X_test,Y_test):
    predictions = SklearnSupervisedLearning(X_train,Y_train,X_test)
    for supervisedmodel in predictions:
        ConfusionMatrix(Y_test, supervisedmodel)


def gridsearchSVC():
    
    
    param_grid = {'C': [1, 10, 100],'gamma': [1, 0.1, 0.01],'kernel': ['linear']}  
    grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3) 
    result = grid.fit(X_train, y_train)
    print(result.best_score_)
    print(result.best_params_)
    #result.cv_results_
    Accuracy = [0.9283, 0.9283, 0.9283, 0.9313, 0.9313, 0.9313, 0.9331, 0.9331, 0.9331]
    C = [1,1,1,10,10,10,100,100,100]
    plt.plot(Accuracy , C)
    plt.xlabel("Accuracy")
    plt.ylabel("C values")
    plt.title("Hyperparameter search for SVM")
    
def gridsearchDecisionTree():
    
    param_grid = { 'criterion':['gini','entropy'],'max_depth': np.arange(2, 7)}
    grid = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=2)
    result = grid.fit(X_train, y_train)
    print(result.best_score_)
    print(result.best_params_)
    #result.cv_results_
    Accuracy = [0.27445335, 0.36556122, 0.52521866, 0.69329446, 0.80255102,0.36497813, 0.67026239, 0.8127551 , 0.85466472, 0.90058309]
    max_depth =[2, 3, 4, 5, 6, 2, 3, 4, 5, 6]
    plt.plot(Accuracy , max_depth)
    plt.xlabel("Accuracy")
    plt.ylabel("Max Depth")
    plt.title("Hyperparameter search for Decision Tree")
    
def gridsearchKNN():
    
    param_grid = {'n_neighbors':[4,5,6],'leaf_size':[1,3,5],'algorithm':['auto', 'kd_tree'],'n_jobs':[-1]}
    grid = GridSearchCV(KNeighborsClassifier(), param_grid=parameters)
    result = grid.fit(train_X,train_y)
    print(result.best_score_)
    print(result.best_params_)
    plt.plot(result.best_score_)
    neighbours = [4, 5, 4, 5, 4, 5]
    Accuracy = [0.75798105, 0.76603499, 0.75798105, 0.76603499, 0.75798105,0.76603499]
    plt.plot(Accuracy , neighbours)
    plt.xlabel("Accuracy")
    plt.ylabel("Neighbours")
    plt.title("Hyperparameter search for KNN")


    
