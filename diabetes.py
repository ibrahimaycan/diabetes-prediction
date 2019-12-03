# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 18:17:58 2019

@author: AYCAN
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#Data preprocessing
diabetesDataset= pd.read_csv('diabetes.csv')
correlation=diabetesDataset.corr()

#eksik veriler

(diabetesDataset.Pregnancies == 0).sum(),(diabetesDataset.Glucose==0).sum(),(diabetesDataset.BloodPressure==0).sum(),(diabetesDataset.SkinThickness==0).sum(),(diabetesDataset.Insulin==0).sum(),(diabetesDataset.BMI==0).sum(),(diabetesDataset.DiabetesPedigreeFunction==0).sum(),(diabetesDataset.Age==0).sum()
## Counting cells with 0 Values for each variable and publishing the counts below
##(111, 5, 35, 227, 374, 11, 0, 0)


######################################0ların drop edilmiş hali

drop_Glu=diabetesDataset.index[diabetesDataset.Glucose == 0].tolist()
drop_BP=diabetesDataset.index[diabetesDataset.BloodPressure == 0].tolist()
drop_Skin = diabetesDataset.index[diabetesDataset.SkinThickness==0].tolist()
drop_Ins = diabetesDataset.index[diabetesDataset.Insulin==0].tolist()
drop_BMI = diabetesDataset.index[diabetesDataset.BMI==0].tolist()
c=drop_Glu+drop_BP+drop_Skin+drop_Ins+drop_BMI
dia=diabetesDataset.drop(diabetesDataset.index[c])
dia0 = dia[dia.Outcome==0]## all variables with 0 outcome
dia1 = dia[dia.Outcome==1]## all variables with 1 outcome

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




#####################################################################

missingValues=diabetesDataset.iloc[:,1:6].values
missingValuesDF=pd.DataFrame(data=missingValues,index=range(768),columns=['Glucose',
                          'BloodPressure','SkinThickness','Insulin','BMI'])
missingValuesDF=missingValuesDF.replace(0,np.nan)
missingValues=missingValuesDF.iloc[:,0:5].values
from sklearn.preprocessing import Imputer
imputer= Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer=imputer.fit(missingValues[:,0:5])
missingValues[:,0:5] = imputer.transform(missingValues[:,0:5])
print(missingValues)
##

fig, ax = plt.subplots()