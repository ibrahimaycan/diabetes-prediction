# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 18:17:58 2019

@author: AYCAN
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

global x_train, x_test, y_train, y_test, diabetesDataset, model_Input_DF, outcome_DF


def processingDataset():

    global diabetesDataset
    # Data preprocessing
    diabetesDataset = pd.read_csv('diabetes.csv')
    # correlation = diabetesDataset.corr()

    # eksik verileri sayıyor
    (diabetesDataset.Pregnancies == 0).sum(), (diabetesDataset.Glucose == 0).sum(),
    (diabetesDataset.BloodPressure == 0).sum(), (diabetesDataset.SkinThickness == 0).sum(),
    (diabetesDataset.Insulin == 0).sum(), (diabetesDataset.BMI == 0).sum(),
    (diabetesDataset.DiabetesPedigreeFunction == 0).sum(), (diabetesDataset.Age == 0).sum()
    # Counting cells with 0 Values for each variable and publishing the counts below
    # (111, 5, 35, 227, 374, 11, 0, 0)

    # Missing valueları ortalama ile degistiriyor
    pregnancies = diabetesDataset.iloc[:, 0:1].values
    pregnanciesDF = pd.DataFrame(data=pregnancies, index=range(768), columns=['Pregnancies'])
    DPFandAge = diabetesDataset.iloc[:, 6:-1].values
    DPFandAgeDF = pd.DataFrame(data=DPFandAge, index=range(768), columns=['DiabetesPedigreeFunction', 'Age'])
    missingValues = diabetesDataset.iloc[:, 1:6].values
    missingValuesDF = pd.DataFrame(data=missingValues, index=range(768), columns=['Glucose',
                                                                                  'BloodPressure', 'SkinThickness',
                                                                                  'Insulin', 'BMI'])
    # missing valueları NaN yapıyor
    missingValuesDF = missingValuesDF.replace(0, np.nan)
    diabetesDatasetRevized = pd.concat([pregnanciesDF, missingValuesDF], axis=1)
    diabetesDatasetRevized = pd.concat([diabetesDatasetRevized, DPFandAgeDF], axis=1)

    missingValues = diabetesDatasetRevized.iloc[:, :].values
    from sklearn.preprocessing import Imputer

    imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imputer = imputer.fit(missingValues[:, :])
    missingValues[:, :] = imputer.transform(missingValues[:, :])
    print(missingValues)

    # verilerin train ve test olarak bölünmesi
    outcome = diabetesDataset.iloc[:, -1:].values
    outcome_DF = pd.DataFrame(data=outcome, index=range(768), columns=['Outcome'])

    model_Input_DF = pd.DataFrame(data=missingValues, index=range(768),
                                  columns=['Pregnancies', 'Glucose', 'BloodPressure',
                                           'SkinThickness', 'Insulin', 'BMI',
                                           'DiabetesPedigreeFunction', 'Age'])

    return model_Input_DF, outcome_DF



def linearRegression():

    global x_test, y_test, x_train, y_train
    regressor = LinearRegression()
    regressor.fit(x_train, y_train)
    y_predict = regressor.predict(x_test)
    # y_testValues = y_test.iloc[:, :].values
    print("Linear Regression accuracy  ", metrics.accuracy_score(y_test, y_predict.round()))


def backwardEliminationLinearRegression():

    global model_Input_DF
    X = np.append(arr=np.ones((768, 1)).astype(int), values=model_Input_DF, axis=1)
    X_l = model_Input_DF.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7]].values
    r_ols = sm.OLS(endog=outcome_DF, exog=X_l)
    r = r_ols.fit()
    print(r.summary())

    # En yuksek p degeri 7. kolonda o yüzden onu eledik
    X_l = model_Input_DF.iloc[:, [0, 1, 2, 3, 4, 5, 6]].values
    r_ols = sm.OLS(endog=outcome_DF, exog=X_l)
    r = r_ols.fit()
    print(r.summary())

    # En yuksek p degeri 4. kolonda o yüzden onu eledik
    X_l = model_Input_DF.iloc[:, [0, 1, 2, 3, 5, 6]].values
    r_ols = sm.OLS(endog=outcome_DF, exog=X_l)
    r = r_ols.fit()
    print(r.summary())

    # En yuksek p degeri 3. kolonda o yüzden onu eledik
    X_l = model_Input_DF.iloc[:, [0, 1, 2, 5, 6]].values
    r_ols = sm.OLS(endog=outcome_DF, exog=X_l)
    r = r_ols.fit()
    print(r.summary())

    model_Input_DF = pd.DataFrame(data=X_l, index=range(768), columns=['Pregnancies', 'Glucose', 'BloodPressure', 'BMI',
                                                                       'DiabetesPedigreeFunction'])

    x_train_lr, x_test_lr, y_train_lr, y_test_lr = train_test_split(model_Input_DF, outcome_DF,
                                                        test_size=0.33, random_state=96)

    regressor = LinearRegression()
    regressor.fit(x_train_lr, y_train_lr)
    y_predict = regressor.predict(x_test_lr)
    y_testValues = y_test.iloc[:, :].values
    print("Linear Regression with backward Elimination: ", metrics.accuracy_score(y_testValues, y_predict.round()))



def svm():

    global x_train, y_train, x_test, y_test

    clf = SVC(kernel='linear')
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print("SVM Accuracy : ", accuracy_score(y_test, y_pred))


def decision_tree():

    global x_train, y_train, x_test, y_test
    clf = DecisionTreeClassifier()
    clf = clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print("Decision Tree Accuracy : ", metrics.accuracy_score(y_test, y_pred))


def k_neighbors_classifier(neighbor):

    global x_train, y_train, x_test, y_test
    knn = KNeighborsClassifier(n_neighbors=neighbor, metric='euclidean')
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    print("K-Nearest Neighbor Accuracy : ", metrics.accuracy_score(y_test, y_pred))


def gaussian_nb():

    global x_train, y_train, x_test, y_test
    NB = GaussianNB()
    NB.fit(x_train, y_train)
    y_pred = NB.predict(x_test)
    print(" Naive Bayes Accuracy : ", metrics.accuracy_score(y_test, y_pred))


def random_forest_regressor():

    global x_train, y_train, x_test, y_test
    rf = RandomForestRegressor(n_estimators=50, random_state=0)
    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_test)
    print("Random Forest Regressor : ", metrics.accuracy_score(y_test, y_pred.round()))


def gradient_boosting_classifier():

    global x_train, y_train, x_test, y_test
    gb_clf2 = GradientBoostingClassifier(n_estimators=20, learning_rate=0.5, max_features=2, max_depth=2, random_state=0)
    gb_clf2.fit(x_train, y_train)
    y_pred = gb_clf2.predict(x_test)

    print("Gradient Boosting Classifier : ", metrics.accuracy_score(y_test, y_pred))

model_Input_DF, outcome_DF = processingDataset()
x_train, x_test, y_train, y_test = train_test_split(model_Input_DF, outcome_DF,
                                                test_size=0.33, random_state=96)
'''
linearRegression()
backwardEliminationLinearRegression()
svm()
decision_tree()
'''

k_neighbors_classifier(23)

'''
gaussian_nb()
random_forest_regressor()
gradient_boosting_classifier()
'''
