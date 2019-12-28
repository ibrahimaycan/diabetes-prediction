# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 18:17:58 2019

@author: AYCAN
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Calculating success rate
def calculateSuccessRate(y_predictVal, y_actualVal):
    matched = 0
    for x in range(y_predictVal.size):
        if y_predictVal[x] > 0.5:
            if y_actualVal[x] == 1:
                matched += 1

        else:
            if y_actualVal[x] == 0:
                matched += 1
    success_rate = matched / y_predict.size
    return success_rate


# Data preprocessing
diabetesDataset = pd.read_csv('diabetes.csv')
correlation = diabetesDataset.corr()

# eksik verileri sayıyor
(diabetesDataset.Pregnancies == 0).sum(), (diabetesDataset.Glucose == 0).sum(),
(diabetesDataset.BloodPressure == 0).sum(), (diabetesDataset.SkinThickness == 0).sum(),
(diabetesDataset.Insulin == 0).sum(), (diabetesDataset.BMI == 0).sum(),
(diabetesDataset.DiabetesPedigreeFunction == 0).sum(), (diabetesDataset.Age == 0).sum()
# Counting cells with 0 Values for each variable and publishing the counts below
# (111, 5, 35, 227, 374, 11, 0, 0)

# Missing valueları ortalama ile değiştiriyor
pregnancies = diabetesDataset.iloc[:, 0:1].values
pregnanciesDF = pd.DataFrame(data=pregnancies, index=range(768), columns=['Pregnancies'])
DPFandAge = diabetesDataset.iloc[:, 6:-1].values
DPFandAgeDF = pd.DataFrame(data=DPFandAge, index=range(768), columns=['DiabetesPedigreeFunction', 'Age'])
DPFandAge = diabetesDataset.iloc[:, 6:-1].values
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

model_Input_DF = pd.DataFrame(data=missingValues, index=range(768), columns=['Pregnancies', 'Glucose', 'BloodPressure',
                                                                             'SkinThickness', 'Insulin', 'BMI',
                                                                             'DiabetesPedigreeFunction', 'Age'])

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(model_Input_DF, outcome_DF,
                                                    test_size=0.33, random_state=0)

# linear regression
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train, y_train)
y_predict = regressor.predict(x_test)
y_testValues = y_test.iloc[:, :].values
successRate = calculateSuccessRate(y_predict, y_testValues)
print("Success rate of linear Regression is  " + str(successRate))

# linear regression using backward elimination
import statsmodels.api as sm

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

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(model_Input_DF, outcome_DF,
                                                    test_size=0.33, random_state=0)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train, y_train)
y_predict = regressor.predict(x_test)
y_testValues = y_test.iloc[:, :].values
successRate = calculateSuccessRate(y_predict, y_testValues)
print("Success rate of linear Regression with backward Elimination is  " + str(successRate))

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(x_train)
y = sc_y.fit_transform(y_train)

'''from sklearn.svm import SVR

regressor = SVR(kernel='rbf')
regressor.fit(X, y)

y_pred = regressor.predict(np.array([11]).reshape(1, -1))

'''
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

clf = SVC(kernel='linear')
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print("SVM Accuracy : ", accuracy_score(y_test, y_pred))

from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

clf = DecisionTreeClassifier()
clf = clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print("Decision Tree Accuracy : ", metrics.accuracy_score(y_test, y_pred))

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=101, metric='euclidean')
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
print("K-Nearest Neighbor Accuracy : ", metrics.accuracy_score(y_test, y_pred))

from sklearn.naive_bayes import GaussianNB

NB = GaussianNB()
NB.fit(x_train, y_train)
y_pred = NB.predict(x_test)
print(" Naive Bayes Accuracy : ", metrics.accuracy_score(y_test, y_pred))
