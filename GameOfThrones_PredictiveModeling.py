# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 04:14:34 2019

@author: Thitiwat W. (Win)
"""
# Loading Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.api as smc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

# loading data
GOT = pd.read_excel("cleaned_GOT.xlsx")

######################################
#  Data Analysis ##
######################################

# after run all numberical variables and check p-values from statmodels
# preparing a DataFrame based the the analysis above
GOT_data = GOT.loc[:, [
                    'book1_A_Game_Of_Thrones',
                    'book4_A_Feast_For_Crows',
                    'great_house',
                    'numDeadRelations',
                    'popularity',
                    'm_house'
                    ]]
# preparing the target variable
GOT_target = GOT.loc[:, "isAlive"]

# Train test split with random_state at 508 and test size 10%
X_train, X_test, y_train, y_test = train_test_split(GOT_data,
                                                    GOT_target,
                                                    test_size=0.1,
                                                    random_state=508,
                                                    stratify=GOT_target)

# run Logit to get statistic summary of a model
got_smc = smc.Logit(GOT_target, GOT_data)
results = got_smc.fit()
results.summary()

#############################################
# hyperparameter tuning in scikit-learn
#############################################

# after getting all needed variables, it's time to try on a classifier model
# initialize kneighborsclassifier
knn = KNeighborsClassifier()

# fitting the model
knn.fit(X_train, y_train)

knn_optimal_pred_train = knn.predict(X_train)
knn_optimal_pred_test = knn.predict(X_test)

# predictions
y_pred = knn.predict(X_test)

# Let's compare the testing score to the training score.
print('Training Score', knn.score(X_train, y_train).round(4))
print('Testing Score:', knn.score(X_test, y_test).round(4))

# Let's compare the AUC testing score to the training score.
print(
    'Training AUC Score',
    roc_auc_score(y_train, knn_optimal_pred_train).round(4))
print(
    'Testing AUC Score',
    roc_auc_score(y_test, knn_optimal_pred_test).round(4))

print(confusion_matrix(y_true=y_test,
                       y_pred=y_pred))

# Let's optimize our hyperparameters using GridSearchCV

# Creating a hyperparameter grid
grid_params = {
    'n_neighbors': np.arange(1, 30)
}

# Building the model object one more time
knn_object = KNeighborsClassifier()

# Creating a GridSearchCV object
knn_gs = GridSearchCV(
    knn_object,
    grid_params,
    return_train_score=False,
    scoring=None,
    cv=3
    )

# Fitting the model based on the training data
knn_gs_results = knn_gs.fit(X_train, y_train)

# check results
print(knn_gs_results.best_score_)
print(knn_gs_results.best_estimator_)
print(knn_gs_results.best_params_)

# build model again best on the best param
bp = knn_gs_results.best_params_
knn = KNeighborsClassifier(n_neighbors=bp["n_neighbors"])

knn.fit(X_train, y_train)

knn_optimal_pred_train = knn.predict(X_train)
knn_optimal_pred_test = knn.predict(X_test)

print('Training Score', knn.score(X_train, y_train))
print('Testing Score:', knn.score(X_test, y_test))

print(
    'Training AUC Score',
    roc_auc_score(y_train, knn_optimal_pred_train).round(4))
print(
    'Testing AUC Score',
    roc_auc_score(y_test, knn_optimal_pred_test).round(4))

#############################################
# cross-validation
#############################################
cv_knn_optimal_3 = cross_val_score(
                                knn,
                                GOT_data,
                                GOT_target,
                                cv=3
                                )

print(cv_knn_optimal_3)

print('\nAverage: ',
      pd.np.mean(cv_knn_optimal_3).round(3),
      '\nMinimum: ',
      min(cv_knn_optimal_3).round(3),
      '\nMaximum: ',
      max(cv_knn_optimal_3).round(3))


print(confusion_matrix(y_true=y_test,
                       y_pred=knn_optimal_pred_test))

# Visualizing the confusion matrix again after cross-validation
labels = ['Not Alive', 'Alive']

cm = confusion_matrix(y_true=y_test,
                      y_pred=knn_optimal_pred_test)

sns.heatmap(cm,
            annot=True,
            xticklabels=labels,
            yticklabels=labels,
            cmap='Blues')


plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion matrix of the classifier')
plt.show()

# cross-validation score with 3 folds:
print("cross-validation score with 3 folds",
      pd.np.mean(cv_knn_optimal_3).round(3))

knn_optimal_pred_test
df_knn_pred = pd.DataFrame(
                        knn_optimal_pred_test.round(3),
                        columns=["predicted_alive"])
df_knn_pred.to_excel("predicted_alive.xlsx", index=False)
