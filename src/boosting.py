# coding: utf-8


from utils import *
# ### 3. Boosting

# In[43]:

from sklearn.ensemble import AdaBoostClassifier
# use weak learner = dt
wk_learner_pima = tree.DecisionTreeClassifier(criterion = 'gini', max_depth= 2)
clf_adb = AdaBoostClassifier(wk_learner_pima)

# Make grid to search for optimal n_estimators (max of estimators at chich the algo stops) and learning_rate : 
# the contribution of each classifier by learning rate.
# There is a subtil trade off between those two parameters. Using a grid to find the optimal kopple is useful.

max_nestimators = 200
max_nlrate = 5
range_nestimators = np.arange(1, max_nestimators , 20)
range_nlrate = np.linspace(0.1, 2, max_nlrate)

parameter_grid = {'n_estimators' : range_nestimators,
                  'learning_rate' : range_nlrate}

grid = GridSearchCV(clf_adb,param_grid = parameter_grid,
                           cv = cv)

grid.fit(X1, y1)
pima_boo_score = grid.best_score_
print('Best score: {}'.format(pima_boo_score))
print('Best parameters: {}'.format(grid.best_params_))


# In[45]:

plt.figure()

scores = [x[1] for x in grid.grid_scores_]
scores = np.array(scores).reshape(len(range_nlrate), len(range_nestimators))

for ind, i in enumerate(range_nlrate):
    plt.plot(range_nestimators, scores[ind], label='nlrate: ' + str(i))
plt.legend()
plt.xlabel('n estimators')
plt.ylabel('Mean score')
plt.savefig('./output/grid_search_pima_boo')
plt.show()


# In[49]:

wk_learner_wine = tree.DecisionTreeClassifier(criterion = 'gini', max_depth= 15)
clf_adb = AdaBoostClassifier(wk_learner_wine)

grid = GridSearchCV(clf_adb,param_grid = parameter_grid,
                           cv = cv)
grid.fit(X2_up, y2_up)
wine_boo_score = grid.best_score_
print('Best score: {}'.format(wine_boo_score))
print('Best parameters: {}'.format(grid.best_params_))

plt.figure()

scores = [x[1] for x in grid.grid_scores_]
scores = np.array(scores).reshape(len(range_nlrate), len(range_nestimators))

for ind, i in enumerate(range_nlrate):
    plt.plot(range_nestimators, scores[ind], label='nlrate: ' + str(i))
plt.legend()
plt.xlabel('n estimators')
plt.ylabel('Mean score')
plt.savefig('./output/grid_search_wine_boo15')
plt.show()


# First graph shows some kind of overfitting, after basically n_estimatore = 23 , all curves drop (more or less).
# This is probably due to the size of our Pima dataset. Even if boosting algorithm doesnt tend to overfit, on very
# small dataset, 21 estimators seems to be a limit after which adding tree with max_depth of 2 are just adding some
# useless complexity and reduces the score. 

# Second graphs, however shows a more real trend of Boosting. Indeed, on the wine dataset, scores tend to increase
# proportionally with the number of n estimators to take into consideration.

# Nevertheless, in both cases boosting score is way better (almost 80%) than simply applying a single weak learner.
