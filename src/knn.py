
from utils import *
from sklearn.neighbors import KNeighborsClassifier


# In[35]:

# let try with a k = 1
k = 1
clf_knn = KNeighborsClassifier(n_neighbors=k)


# In[36]:

#X1_train or y1_train ?
title =  'Learning Curve - KNN - pima  '
plt, best_score = plot_learning_curve(clf_knn, title, X1_nor, y1, ylim=(0.3, 1.01), cv=cv, n_jobs=4)
plt.show()
title =  'Learning Curve - KNN - wine  '
plt, best_score = plot_learning_curve(clf_knn, title, X2_nor, y2, ylim=(0.3, 1.01), cv=cv, n_jobs=4)
plt.show()


# In[53]:

# finding K
# for pima
title = " Validation Curve - K - pima "
xlabel = "K neighbors"
ylabel = "Score"
k_range = np.arange(1,30,1)
valid_curve_knn_pima, pima_knn_score, best_param = plot_validation_curve(clf_knn, title, xlabel, ylabel,X1_nor, y1, param_name = 'n_neighbors', ylim=None, 
                              cv = cv, n_jobs = 1, param_range = k_range)
valid_curve_knn_pima.savefig('./output/valid-curve-pima-knn.png')
valid_curve_knn_pima.show()
print("Best score for pima is " + str(pima_knn_score) + ", K = " + str(best_param))

# for wine
title = " Validation Curve - K - pima "
valid_curve_knn_wine, wine_knn_score, best_param = plot_validation_curve(clf_knn, title, xlabel, ylabel,X2_nor, y2, param_name = 'n_neighbors', ylim=None, 
                              cv = cv, n_jobs = 1, param_range = k_range)
valid_curve_knn_wine.savefig('./output/valid-curve-wine-knn.png')
valid_curve_knn_wine.show()
print("Best score for wine is " + str(wine_knn_score) + ", K = " + str(best_param))


# classical example of overfitting . The model has 0 error on Training error : makes sense since every instance's first neighbor is itself. So there is obviously no error.
# 
# However it doesnt perform well on cross val data.
# 
# dimension 