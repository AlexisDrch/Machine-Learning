# ### 4. Support Vector Machines

# In[29]:
from utils import *
from sklearn import svm

# normalize because SVM uses distance as a metric to compute
clf_svm = svm.SVC()

cv_svm = StratifiedKFold(n_splits=10)


# SVM is relatively robust but to better our result we should use a grid to find the best tuned parameters. Kernels and 
# related degree are parameters to tunes.

# In[30]:

clf_svm = svm.SVC()
max_degree = 6
kernel_choices = ['linear', 'sigmoid', 'poly', 'rbf']
degree_choices = np.arange(1, max_degree)
gamma_choices = [1e-1, 1, 1e1]
C_choices = [1e-2, 1, 1e2]

parameters_grid = {
    'kernel' : kernel_choices,
    'degree' : degree_choices,
    'gamma' : gamma_choices,
    'C': [1e-2, 1, 1e2]
}
    
grid = GridSearchCV(
    clf_svm,
    param_grid = parameters_grid,
    cv = cv_svm
)  

print()
grid.fit(X1_nor, y1)
pima_svm_score = grid.best_score_
print('Best score: {}'.format(pima_svm_score))
print('Best parameters: {}'.format(grid.best_params_))


# This result fits our first expectation : the pima dataset is not linearly separable. Instead the best model use a rbf (radial basis function) to maximise the margin between the classes. If we could visualize this multidimensional dataset we would eventually see a boundary with a 'circle shape' between the two classes. The High C and High gamma corresponds to the expectation we have when analysing the dataset : the intersection between the classes is relatively frequent and so the boundray has a complexe shape. The best svm needs to consider more instance as support vectors (high C) and gives more influence to the instances close to the boundary (high gamma).

# In[31]:

clf_svm = svm.SVC()
max_degree = 6
kernel_choices = ['linear', 'sigmoid', 'poly', 'rbf']
degree_choices = np.arange(1, max_degree)
gamma_choices = [1e-1, 1, 1e1]
C_choices = [1e-2, 1, 1e2]

parameters_grid = {
    'kernel' : kernel_choices,
    'degree' : degree_choices,
    'gamma' : gamma_choices,
    'C': [1e-2, 1, 1e2]
}
                  

grid = GridSearchCV(
    clf_svm,
    param_grid = parameters_grid,
    cv = cv_svm
)  

grid.fit(X2_nor, y2)
wine_svm_score = grid.best_score_
print('Best score: {}'.format(wine_svm_score))
print('Best parameters: {}'.format(grid.best_params_))



# ### 3. iteration curve: since svm is an iteration algorithm

# In[50]:

# degree can be ignoer out of poly
max_iter = 10000 # set really large to see where we should stop
title_pima = "Iteration on pima dataset with optimal svm"
svm_pima = svm.SVC(C = 100.0, gamma = 10.0, kernel = 'rbf')
plot = plot_iterative_learning_curve(svm_pima, title_pima, X1_nor, y1, ylim=None, cv = cv_svm, n_jobs=-1,
                              iterations=np.arange(1, max_iter, 1000))
plot.savefig('./output/iterative-pima-svm.png')
plot.show()


# In[51]:

# degree can be ignoer out of poly
max_iter = 10000 # set really large to see where we should stop
title_pima = "Iteration on wine dataset with optimal svm"
svm_wine = svm.SVC(C = 100.0, gamma = 10.0, kernel = 'sigmoid')
plot = plot_iterative_learning_curve(svm_wine, title_pima, X2_nor, y2, ylim=None, cv = cv_svm, n_jobs=-1,
                              iterations=np.arange(1, max_iter, 1000))
plot.savefig('./output/iterative-wine-svm.png')
plot.show()


# Only around 1800 iterations is required to reach the max score. Then no more overfitting but not interest : loose of time.
