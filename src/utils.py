# coding: utf-8


# In[1]:

import pandas as pd
import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV 
from sklearn.preprocessing import normalize
from sklearn.utils import resample 
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.model_selection import ShuffleSplit

from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition
         

# In[2]:

# Retrieving data and basic checks
df1 = pd.read_csv('../data/Pima.csv')
assert df1.isnull().sum().sum() == 0 # check that there is no null value
df1.head(n = 5)


# In[3]:

# Retrieving data and basic checks
df2 = pd.read_csv('../data/winequality-red.csv', sep= ';')
assert df2.isnull().sum().sum() == 0 # check that there is no null value
df2.head(n = 5)


# In[4]:

# 1st classification problems : Pima binary classification probs
# Building subsets
X1 = df1.iloc[:,0:-1]
y1 = df1.iloc[:,-1]

[m1, n1] = X1.shape
print('1st DataFrame:\n' +
     ' - Dimension n1 = {}\n'.format(n1) +
     ' - Size m1 = {}\n'.format(m1) +
     ' - type of features = {}'.format('numericals / continuous & discrete'))

# Check for highly correlated features
Xtemp = df1.corr().unstack().reset_index()
Xtemp[(Xtemp[0]>0.5) & (Xtemp[0] != 1)]

# In[7]:

# Compare covariance and mean (distribution of data from class 1 is the same as data from class2)
mu1 = X1[y1 == 1].mean()
print(mu1)
print('p1 = ' + str(X1[y1 == 1].shape[0]/m1))
cov1 = X1[y1 == 1].cov()
cov1[cov1 < 30] = 0
cov1


# In[8]:

mu2 = X1[y1 == 2].mean()
print(mu2)
print('p2 = ' + str(X1[y1 == 2].shape[0] / m1))
cov2 = X1[y1 == 2].cov()
cov2[cov2 < 30] = 0
cov2


# In[10]:

# 2nd classification problems : 
# Building subsets
X2 = df2.iloc[:,0:-1]
y2 = df2.iloc[:,-1]

[m2, n2] = X2.shape
print('2nd DataFrame:\n' +
     ' - Dimension n2 = {}\n'.format(n2) +
     ' - Size m2 = {}\n'.format(m2) +
     ' - type of features = {}'.format('numericals / continuous & discrete'))


# In[11]:
# correlation check
Xtemp = df2.corr().unstack().reset_index()
Xtemp[(Xtemp[0]>0.5) & (Xtemp[0] != 1)]


# In[14]:

df23 = df2[df2.quality == 3]
df24 = df2[df2.quality == 4]
df25 = df2[df2.quality == 5]
df26 = df2[df2.quality == 6]
df27 = df2[df2.quality == 7]
df28 = df2[df2.quality == 8]


# In[15]:

# Upsambpling minority class

# Upsample minority class
df23_upsampled = resample(df23,replace=True,     # sample with replacement
                            n_samples=400,    # to match majority class
                            random_state=123) # reproducible results

# Upsample minority class
df24_upsampled = resample(df24,replace=True,     # sample with replacement
                            n_samples=576,    # to match majority class
                            random_state=123) # reproducible results

# Upsample minority class
df27_upsampled = resample(df27,replace=True,     # sample with replacement
                            n_samples=576,    # to match majority class
                            random_state=123) # reproducible results

# Upsample minority class
df28_upsampled = resample(df28,replace=True,     # sample with replacement
                            n_samples=400,    # to match majority class
                            random_state=123) # reproducible results
 
# Combine majority class with upsampled minority class
df2_upsampled = pd.concat([df23_upsampled, df24_upsampled, df25, df26, df27_upsampled, df28_upsampled])
 
# Display new class counts
df2_upsampled.quality.value_counts()


# In[16]:

X2_up = df2_upsampled.iloc[:,0:-1]
y2_up = df2_upsampled.iloc[:,-1]


# In[17]:

X1_nor = normalize(X1, axis = 0)
X2_nor = normalize(X2, axis = 0)
X2_up_nor = normalize(X2_up, axis = 0)

cv = StratifiedKFold(n_splits=10)


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    best_test_scores = np.max(test_scores_mean)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt, best_test_scores
    
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.

def clean(X1):
    # Check and remove useless columns
    for col in X1:
        if(len((X1[col].unique())) == 1):
            print("Feature '{}' has a unique value for all the data = {}".format(col, X1[col].unique()))
            X1 = X1.drop([col], axis = 1)
            print("Feature '{}' has been removed ".format(col))

    # One hot encode for categorical features
    for col in X1:
        one_hot_col = pd.get_dummies(X1[col], prefix=col)
        X1 = X1.drop([col], axis = 1)
        X1 = X1.join(one_hot_col)
        
    return X1

def plot_PCA_3(X,y, dic):

    np.random.seed(5)

    centers = [[1, 1], [-1, -1], [1, -1]]

    fig = plt.figure(1, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

    plt.cla()
    pca = decomposition.PCA(n_components=3)
    pca.fit(X)
    X = pca.transform(X)

    for name, label in dic:
        ax.text3D(X[y == label, 0].mean(),
                  X[y == label, 1].mean() + 1.5,
                  X[y == label, 2].mean(), str('cat_' + name),
                  horizontalalignment='center',
                  bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
    # Reorder the labels to have colors matching the cluster results
    y = np.choose(y, [1,2, 0]).astype(np.float)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.spectral,
               edgecolor='k')

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])

    plt.show()
    return

def plot_validation_curve(estimator, title, xlabel, ylabel,X, y,param_name, ylim=None, 
                          cv=None,n_jobs=1, param_range = np.linspace(1, 1, 10)):
   
    train_scores, test_scores = validation_curve(
        estimator, X, y, param_name= param_name, param_range=param_range,
        cv=cv, scoring= "accuracy", n_jobs = n_jobs)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    best_test_scores = np.max(test_scores_mean)
    best_param = np.argmax(test_scores_mean)
    
    if ylim is not None:
        plt.ylim(*ylim)
    lw = 2
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(param_range, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.plot(param_range, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    return plt, best_test_scores, best_param


def plot_iterative_learning_curve(estimator, title, X, y, iterations, ylim=None, cv=None, n_jobs=-1):
    
    plt.figure(figsize=(10, 10))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Iterations")
    plt.ylabel("Score")

    parameter_grid = {'max_iter': iterations}
    grid_search = GridSearchCV(estimator, param_grid=parameter_grid, n_jobs=-1, cv=cv)
    grid_search.fit(X, y)

    train_scores_mean = grid_search.cv_results_['mean_train_score']
    train_scores_std = grid_search.cv_results_['std_train_score']
    test_scores_mean = grid_search.cv_results_['mean_test_score']
    test_scores_std = grid_search.cv_results_['std_test_score']
    plt.grid()

    plt.fill_between(iterations, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(iterations, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(iterations, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(iterations, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt