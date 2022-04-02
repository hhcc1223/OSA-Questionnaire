# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 00:00:18 2021

@author: Jiayan
"""

#import package
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
from scipy import stats
from collections import Counter

#settings
my_path = os.path.abspath(os.path.dirname(__file__))
data_path = os.path.join(my_path, "cleaned_shhs.csv")
np.random.seed(10)
wsc_path = os.path.join(my_path, "wsc_testing_new.csv")

#Load the dataset
wsc_all = pd.read_csv(wsc_path)
all_shhs1 = pd.read_csv(data_path)

all_shhs1['LoudSn02'] = all_shhs1['LoudSn02'].fillna(0)
all_shhs1['LoudSn02'] = all_shhs1['LoudSn02'].replace(8,0)
all_shhs1['HOSnr02'] = all_shhs1['HOSnr02'].fillna(1)
all_shhs1['HOSnr02'] = all_shhs1['HOSnr02'].replace(8,1)
all_shhs1['Drive02'] = all_shhs1['Drive02'].fillna(1)

wsc_all['LoudSn02'] = wsc_all['LoudSn02'].fillna(0)
wsc_all['LoudSn02'] = wsc_all['LoudSn02'].replace(8,0)
wsc_all['HOSnr02'] = wsc_all['HOSnr02'].fillna(1)
wsc_all['HOSnr02'] = wsc_all['HOSnr02'].replace(8,1)
wsc_all['Drive02'] = wsc_all['Drive02'].fillna(1)


selected_features = ['HOSnr02','LoudSn02','StpBrt02','HTNDerv_s1','gender','ESS_s1','Sleepy02','Drive02',
                     'SystBP','bmi_s1', 'DiasBP', 'NECK20', 'age_s1', 'ahi_a0h3a']
all_shhs1 = all_shhs1[selected_features]
ahi_class = []
for i in range(len(all_shhs1)):
    if all_shhs1.iloc[i]["ahi_a0h3a"] <= 15: ahi_class.append(0)
    else: ahi_class.append(1)

all_shhs1['output'] = ahi_class
y = all_shhs1['output']
all_shhs1 = all_shhs1.drop(columns =['ahi_a0h3a', 'output'])

X_train, X_test, y_train, y_test = train_test_split(
        all_shhs1,y,test_size = 0.3,random_state = 10, shuffle = False)
X_train['output'] = y_train
X_test['output'] = y_test


wsc_all = wsc_all[selected_features]
ahi_class = []
for i in range(len(wsc_all)):
    if wsc_all.iloc[i]["ahi_a0h3a"] <= 15: ahi_class.append(0)
    else: ahi_class.append(1)

wsc_all['output'] = ahi_class
wsc_all['gender'] = wsc_all['gender'].replace(1,2)
wsc_all['gender'] = wsc_all['gender'].replace(0,1)

categorical_feature = ['HOSnr02','LoudSn02','StpBrt02','HTNDerv_s1','gender','ESS_s1','Sleepy02','Drive02', 'output']
continous_feature = ['SystBP','bmi_s1', 'DiasBP', 'NECK20', 'age_s1']
#t-test
for i in range(len(continous_feature)):
    mean_test1 = np.mean(X_test[continous_feature[i]])
    mean_test2 = np.mean(wsc_all[continous_feature[i]])
    std_test1 = np.std(X_test[continous_feature[i]])
    std_test2 = np.std(wsc_all[continous_feature[i]])
    std_all = np.std(list(X_test[continous_feature[i]]) + list(wsc_all[continous_feature[i]]))
    _, p_value = stats.ttest_ind(X_test[continous_feature[i]],wsc_all[continous_feature[i]], equal_var = True)
    print("-" * 20,continous_feature[i] ,"-" * 20)
    print("Mean, std of test1", mean_test1, std_test1)
    print("Mean, std of test2", mean_test2, std_test2)
    print("p-value between test2 and test1", p_value)
    print("Effect size = ", abs(mean_test1 - mean_test2)/std_all)
 
#chi-square test
for i in range(len(categorical_feature)):
    cnt_test1 = Counter(X_test[categorical_feature[i]])
    cnt_test2 = Counter(wsc_all[categorical_feature[i]])
    freq_test1, freq_test2 = [], []
    effect_size = 0
    for x in cnt_test1:
        freq_test1.append(cnt_test1[x])
        freq_test2.append(cnt_test2[x])
        effect_size += (cnt_test2[x] - cnt_test1[x])**2/(cnt_test1[x])
        if cnt_test1[x] * cnt_test2[x] == 0: print(categorical_feature[i], x)
    effect_size = effect_size ** 0.5
    _, p_value = stats.chisquare(freq_test1,freq_test2)
    print("-" * 20,categorical_feature[i] ,"-" * 20)
    print("p-value between test2 and test1", p_value)
    print("Effect size = ", effect_size)



