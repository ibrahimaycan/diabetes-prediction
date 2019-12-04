# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 18:17:58 2019

@author: AYCAN
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

## Calculating success rate
def calculateSuccessRate(y_predictVal,y_actualVal):
    matched=0
    for x in range(y_predictVal.size):
        if y_predictVal[x] > 0.5:
            if y_actualVal[x]==1:
                matched +=1
                
        else:
            if y_actualVal[x]==0:
                matched +=1
    success_rate=matched/y_predict.size
    return success_rate



#Data preprocessing
diabetesDataset= pd.read_csv('diabetes.csv')
correlation=diabetesDataset.corr()

#eksik verileri sayıyor
(diabetesDataset.Pregnancies == 0).sum(),(diabetesDataset.Glucose==0).sum(),
(diabetesDataset.BloodPressure==0).sum(),(diabetesDataset.SkinThickness==0).sum(),
(diabetesDataset.Insulin==0).sum(),(diabetesDataset.BMI==0).sum(),
(diabetesDataset.DiabetesPedigreeFunction==0).sum(),(diabetesDataset.Age==0).sum()
## Counting cells with 0 Values for each variable and publishing the counts below
##(111, 5, 35, 227, 374, 11, 0, 0)

###################################### missing valueların drop edilmiş hali
drop_Glu=diabetesDataset.index[diabetesDataset.Glucose == 0].tolist()
drop_BP=diabetesDataset.index[diabetesDataset.BloodPressure == 0].tolist()
drop_Skin = diabetesDataset.index[diabetesDataset.SkinThickness==0].tolist()
drop_Ins = diabetesDataset.index[diabetesDataset.Insulin==0].tolist()
drop_BMI = diabetesDataset.index[diabetesDataset.BMI==0].tolist()
c=drop_Glu+drop_BP+drop_Skin+drop_Ins+drop_BMI
dia=diabetesDataset.drop(diabetesDataset.index[c])
dia0 = dia[dia.Outcome==0]## all variables with 0 outcome
dia1 = dia[dia.Outcome==1]## all variables with 1 outcome
####################################################

##visualization  on pregnancy
plt.figure(figsize=(20, 6))
plt.subplot(1,3,1)
sns.set_style("dark")
plt.title("Pregnancies")
sns.distplot(dia.Pregnancies,kde=False)
plt.subplot(1,3,2)
sns.distplot(dia0.Pregnancies,kde=False,color="Blue", label="Preg for Outome=0")
sns.distplot(dia1.Pregnancies,kde=False,color = "Gold", label = "Preg for Outcome=1")
plt.title("Histograms for Preg by Outcome")
plt.legend()
plt.subplot(1,3,3)
sns.boxplot(x=dia.Outcome,y=dia.Pregnancies)
plt.title("Boxplot for Preg by Outcome")


#Missing valueları ortalama ile değiştiriyor
#####################################################################
pregnancies = diabetesDataset.iloc[:, 0:1].values
pregnanciesDF = pd.DataFrame(data=pregnancies,index=range(768),columns=['Pregnancies'])
DPFandAge = diabetesDataset.iloc[:, 6:-1].values
DPFandAgeDF = pd.DataFrame(data=DPFandAge, index=range(768), columns=['DiabetesPegidreeFunction', 'Age'])
DPFandAge = diabetesDataset.iloc[:, 6:-1].values
missingValues = diabetesDataset.iloc[:, 1:6].values
missingValuesDF = pd.DataFrame(data = missingValues, index = range(768), columns=['Glucose',
                          'BloodPressure', 'SkinThickness', 'Insulin', 'BMI'])
missingValuesDF = missingValuesDF.replace(0, np.nan) #missing valueları NaN yapıyor
diabetesDatasetRevized = pd.concat([pregnanciesDF, missingValuesDF], axis=1)
diabetesDatasetRevized = pd.concat([diabetesDatasetRevized, DPFandAgeDF], axis=1)


missingValues = diabetesDatasetRevized.iloc[:, :].values
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(missingValues[:, :])
missingValues[:, :] = imputer.transform(missingValues[:, :])


print(missingValues)


#verilerin train ve test olarak bölünmesi
############################################################################
outcome = diabetesDataset.iloc[:, -1:].values
outcome_DF = pd.DataFrame(data=outcome,index=range(768),columns=['Outcome'])

model_Input_DF=pd.DataFrame(data=missingValues, index=range(768), columns=['Pregnancies', 'Glucose', 
                        'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 
                        'DiabetesPedigreeFunction','Age'])
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test= train_test_split(model_Input_DF, outcome_DF, 
                                                   test_size = 0.33, random_state = 0)

#linear regression
from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(x_train,y_train)
y_predict=regressor.predict(x_test)
y_testValues=y_test.iloc[:,:].values
successRate=calculateSuccessRate(y_predict,y_testValues)
print("Success rate of linear Regression is  "+str(successRate))



#linear regression using backward elimination
import statsmodels.formula.api as sm
X=np.append(arr = np.ones((768,1)).astype(int),values= model_Input_DF,axis=1)
X_l=model_Input_DF.iloc[:,[0,1,2,3,4,5,6,7]].values
r_ols=sm.OLS(endog=outcome_DF,exog=X_l)
r=r_ols.fit()
print(r.summary())

X_l=model_Input_DF.iloc[:,[0,1,2,3,4,5,6]].values #En yüksek p değeri 7. kolonda o yüzden onu eledik
r_ols=sm.OLS(endog=outcome_DF,exog=X_l)
r=r_ols.fit()
print(r.summary()) 

X_l=model_Input_DF.iloc[:,[0,1,2,3,5,6]].values #En yüksek p değeri 4. kolonda o yüzden onu eledik
r_ols=sm.OLS(endog=outcome_DF,exog=X_l)
r=r_ols.fit()
print(r.summary()) 

X_l=model_Input_DF.iloc[:,[0,1,2,5,6]].values #En yüksek p değeri 3. kolonda o yüzden onu eledik
r_ols=sm.OLS(endog=outcome_DF,exog=X_l)
r=r_ols.fit()
print(r.summary()) 

model_Input_DF=pd.DataFrame(data=X_l, index=range(768), columns=['Pregnancies', 'Glucose', 
                        'BloodPressure', 'BMI', 
                        'DiabetesPedigreeFunction'])
    
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(model_Input_DF, outcome_DF, 
                                                   test_size = 0.33, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(x_train,y_train)
y_predict=regressor.predict(x_test)
y_testValues=y_test.iloc[:,:].values
successRate=calculateSuccessRate(y_predict,y_testValues)
print("Success rate of linear Regression with backward Elimination is  "+str(successRate))
print()


#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(x_train)
sc2 = StandardScaler()
y_olcekli = sc2.fit_transform(y_train)

from sklearn.svm import SVR

svr_reg = SVR(kernel = 'linear')
svr_reg.fit(x_olcekli,y_olcekli)
y_predict=svr_reg.predict(x_test)


plt.scatter(x_olcekli,y_olcekli,color='red')
plt.plot(x_olcekli,svr_reg.predict(x_olcekli),color='blue')

print(svr_reg.predict(11))









