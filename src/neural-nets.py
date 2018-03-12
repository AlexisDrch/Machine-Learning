# coding: utf-8


from utils import *

# ### 2. Neural Networks

# In[23]:

from sklearn.neural_network import MLPClassifier
clf_nn_adam = MLPClassifier(solver='adam')
clf_nn_lbfgs = MLPClassifier(solver = 'lbfgs') # very heavy to compute (hessian every time)


# In[56]:

# sensibility 
#title = 'Learning Curve - Adam Neural net - wine imbalanced'
#plt, best_score = plot_learning_curve(clf_nn_adam, title, X2, y2, ylim=(0.3, 1.01), cv=cv, n_jobs=4)
#plt.show()

title = 'Learning Curve - Adam Neural net - wine  '
plt, wine_nn_score  = plot_learning_curve(clf_nn_adam, title, X2_up, y2_up, ylim=(0.1, 1.01), cv=cv, n_jobs=4)
plt.savefig('./output/learning-curve-nn-wine.png')
plt.show()
print("wine nn " + str(wine_nn_score))
#title = 'Learning Curve - Quasi Newton Neural net - wine  '
##plt, best_score  = plot_learning_curve(clf_nn_lbfgs, title, X2_up, y2_up, ylim=(0.3, 1.01), cv=cv, n_jobs=4)
#plt.show()

# adam to use : stochastic gradient descent on big dataset : performs better.
# lbfgs : use quasi newton optimization method : performs better on small dataset : ex with Pima.

#title = 'Learning Curve - Adam Neural net - pima  '
#plt, best_score  = plot_learning_curve(clf_nn_adam, title, X1, y1, ylim=(0.3, 1.01), cv=cv, n_jobs=4)
#plt.show()

title = 'Learning Curve - lfbgs Neural net - pima  '
plt, pima_nn_score  = plot_learning_curve(clf_nn_lbfgs, title, X1, y1, ylim=(0.1, 1.01), cv=cv, n_jobs=4)
plt.savefig('./output/learning-curve-nn-pima.png')
plt.show()
print("pima nn " + str(pima_nn_score))


# In[25]:

# pima 
# adam is overfitting
# quasi newton sounds more robust  : cross val score converge towards1  

