"""
Created on Tue Jan 06 13:08:59 2021

@author: Jiayan

for feature analysis
"""

#import package
import pandas as pd
import numpy as np
from sklearn import preprocessing
import os
np.random.seed(10)

def MI_select(my_path):
    data_path = os.path.join(my_path, "cleaned_shhs.csv")
    
    #Load the dataset
    all_subjects = pd.read_csv(data_path)
    
    #Pre-processing: treat don't know as not snoring
    all_subjects['LoudSn02'] = all_subjects['LoudSn02'].fillna(0)
    all_subjects['LoudSn02'] = all_subjects['LoudSn02'].replace(8,0)
    all_subjects['HOSnr02'] = all_subjects['HOSnr02'].fillna(1)
    all_subjects['HOSnr02'] = all_subjects['HOSnr02'].replace(8,1)
    
    #select candidate features
    selected_features = ['nsrrid','LoudSn02', 'bmi_s1', 'gender', 'NECK20', 'age_s1', 'race','HTNDerv_s1', 'STROKE15', 'smokstat_s1', 'ASTHMA15','Alcoh','Sleepy02','ahi_a0h3a']
    all_subjects = all_subjects[selected_features]
    
    #make ground trueth as binary
    ahi_class = []
    for i in range(len(all_subjects)):
        if all_subjects.iloc[i]["ahi_a0h3a"] <= 15: ahi_class.append(0)
        else: ahi_class.append(1)
    
    all_subjects['output'] = ahi_class
    all_subjects = all_subjects.drop(columns =['ahi_a0h3a'])
    
    array = all_subjects.values
    leng = len(all_subjects.columns)
    X = array[:,1:leng-1]
    y = array[:,leng-1]
    features = list(all_subjects.columns)[1:leng-1]
 
    standarlized_X_train = preprocessing.scale(X)
    from sklearn.metrics import normalized_mutual_info_score 
    f_mi2 = {}
    for i in range(len(features)):
        f_mi2[features[i]] = normalized_mutual_info_score(standarlized_X_train[:, i: i+1].reshape((y.shape[0])), y)
    
    ranked_features = sorted(f_mi2.items(), key = lambda x: x[1], reverse=True)
    print("-"*20, "Ranked features based on MI", "-" * 20)
    for i in range(len(ranked_features)):
        print("Feature: %s, NMI: %.3f" %(ranked_features[i][0], ranked_features[i][1]))   
    return f_mi2

if __name__ == "__main__":
    my_path = os.path.abspath(os.path.dirname(__file__))
    f = MI_select(my_path)

