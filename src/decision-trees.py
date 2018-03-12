# coding: utf-8

from utils import *
# ### 1. Decision Trees

# In[18]:


clf_dt = tree.DecisionTreeClassifier(criterion='gini') # explain use of gini and information gain


# #### 1. Balancing wine dataset

# In[19]:

cv = StratifiedKFold(n_splits=10)

title = 'Learning Curve - Decision Tree - wine imbalanced'
plt, score = plot_learning_curve(clf_dt, title, X2, y2, ylim=(0.3, 1.01), cv=cv, n_jobs=4)
plt.savefig('./output/dtree-wine-imbalanced.png')
plt.show()
title = 'Learning Curve - Decision Tree - wine balanced'
plt,score= plot_learning_curve(clf_dt, title, X2_up, y2_up, ylim=(0.3, 1.01), cv=cv, n_jobs=4)
plt.savefig('./output/dtree-wine-balanced.png')
plt.show()


# In[20]:

# on Pima 
title = 'Learning Curve - Decision Tree - pima '
plt,score = plot_learning_curve(clf_dt, title, X1, y1, ylim=(0.3, 1.01), cv=cv, n_jobs=4)
plt.savefig('./output/dtree-pima.png')
plt.show()


# #### 2. Parameters tuning.


# In[42]:

# for pima
max_d = 30
title = " Validation Curve - max_depth - pima "
xlabel = "max_depth"
ylabel = "Score"

clf_dt.fit(X1, y1)
valid_curve_dt_pima, pima_dt_score, best_param = plot_validation_curve(clf_dt, title, xlabel, ylabel,X1, y1, param_name = 'max_depth', ylim=None, 
                              cv = cv, n_jobs = 1, param_range = np.arange(1, max_d))
valid_curve_dt_pima.savefig('./output/valid_curve_dt_pima.png')
valid_curve_dt_pima.show()
print("Best score for pima is " + str(pima_dt_score) + ", max_depth = " + str(best_param))

# for wine
title = " Validation Curve - max_depth - wine "
clf_dt.fit(X2_up, y2_up)
valid_curve_dt_wine, wine_dt_score, best_param = plot_validation_curve(clf_dt, title, xlabel, ylabel,X2_up, y2_up, param_name = 'max_depth', ylim=None, 
                              cv = cv, n_jobs = 1, param_range = np.arange(1, max_d))
valid_curve_dt_wine.savefig('./output/valid_curve_dt_wine.png')
valid_curve_dt_wine.show()
print("Best score for wine is " + str(wine_dt_score) + ", max_depth = " + str(best_param))

