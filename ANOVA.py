# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 13:06:55 2021

@author: apowe
"""
##LINK:https://www.kaggle.com/rashikrahmanpritom/heart-attack-analysis-prediction-dataset?select=heart.csv
##IMPORT
import pandas as pd 
import seaborn as sns
import scipy.stats as stats
import numpy as np 
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import statsmodels.stats.multicomp as mc
import pingouin as pg
from scipy import stats

stats.describe(...)

##IMPORT DF
df = pd.read_csv('C:/Users/apowe/Downloads/heart.csv')
##LIST OF DF FOR ANOVA VARIABLE
list (df)
['age',
 'sex',
 'cp', ##Chest Pain type chest pain type
 'trtbps', ##resting blood pressure (in mm Hg)
 'chol', ##cholestoral in mg/dl fetched via BMI sensor
 'fbs', ##(fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
 'restecg', ##resting electrocardiographic results
 'thalachh', ##maximum heart rate achieved
 'exng', ##exercise induced angina (1 = yes; 0 = no)
 'oldpeak', ##Previous peak
 'slp', ##Slope
 'caa', ##number of major vessels (0-3)
 'thall', ##Thal rate
 'output'] ##Target variable

## VARIABLES FOR 1 WAY ANOVA TESTS 

Dependent variable 1 (continuous value) = restecg
Indepdendent variable 1 (categorical value) = age 
Indepdendent variable 2 (categorical value) = caa
Independent variable  3 (categorical value) = sex
Independent variable  4 (categorical value) = cp
##BOX PLOT
df_age_boxplot = sns.boxplot(x='age', y= 'restecg', data=df, palette="Set3")
df_caa_boxplot = sns.boxplot(x='caa', y= 'restecg', data=df, palette="Set3") 
df_sex_boxplot = sns.boxplot(x='sex', y= 'restecg', data=df, palette="Set3") 
df_cp_boxplot = sns.boxplot(x='cp', y= 'restecg', data=df, palette="Set3") 
##BARPLOTS
df_vs_age = sns.barplot(x='age', y= 'restecg', data=df, palette="Set2") 
df_vs_caa = sns.barplot(x='caa', y= 'restecg', data=df, palette="Set2") 
df_vs_sex = sns.barplot(x='sex', y= 'restecg', data=df, palette="Set2") 
df_vs_cp = sns.barplot(x='cp', y= 'restecg', data=df, palette="Set2") 

##CREATE DF WHERE ONLY COLUMNS NEEDED ARE VISABLE
workingdf = df[['restecg', 'age','caa','sex', 'cp']]
##VALUE COUNTS FOR BALENCED OR UNBALENCED
age_counts = workingdf['age'].value_counts().reset_index()
caa_counts = workingdf['caa'].value_counts().reset_index()
sex_counts = workingdf['sex'].value_counts().reset_index()
cp_counts = workingdf['cp'].value_counts().reset_index()
##1 WAY ANOVA TESTS
model = ols('restecg ~ C(age)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=1)
anova_table
             df     sum_sq   mean_sq         F    PR(>F)
C(age)     40.0  14.312609  0.357815  1.354755  0.085675
Residual  262.0  69.198942  0.264118       NaN       NaN

##RESULT: no signifitcant diffrence or result

model = ols('restecg ~ C(caa)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=1)
anova_table
             df     sum_sq   mean_sq         F    PR(>F)
C(caa)      4.0   1.444287  0.361072  1.311112  0.265822
Residual  298.0  82.067264  0.275394       NaN       NaN

##RESULT: no signifitcant diffrence or result

model = ols('restecg ~ C(sex)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=1)
anova_table
             df     sum_sq   mean_sq         F    PR(>F)
C(sex)      1.0   0.282837  0.282837  1.022893  0.312646
Residual  301.0  83.228714  0.276507       NaN       NaN

##RESULT: no signifitcant diffrence or result

model = ols('restecg ~ C(cp)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=1)
anova_table
             df     sum_sq   mean_sq         F    PR(>F)
C(cp)       3.0   1.669414  0.556471  2.032999  0.109313
Residual  299.0  81.842137  0.273720       NaN       NaN

##RESULT: no signifitcant diffrence or result

##POST COMP TESTS
import statsmodels.stats.multicomp as mc
comp1 = mc.MultiComparison(df['restecg'], df['age'])
post_hoc_res = comp1.tukeyhsd() 
tukeyway1 = post_hoc_res.summary()

comp2 = mc.MultiComparison(df['restecg'], df['caa'])
post_hoc_res2 = comp2.tukeyhsd() 
tukeyway2 = post_hoc_res2.summary()

comp3 = mc.MultiComparison(df['restecg'], df['sex'])
post_hoc_res3 = comp3.tukeyhsd() 
tukeyway3 = post_hoc_res3.summary()

comp4 = mc.MultiComparison(df['restecg'], df['cp'])
post_hoc_res4 = comp4.tukeyhsd() 
tukeyway4 = post_hoc_res4.summary()