# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 01:06:07 2021

@author: gudu0
"""

import os
import pandas as pd
import numpy as np

def shhs_preprocessing(file_path):
    
    shhs_dataset = "shhs1-dataset-0.15.0.csv"
    file_path = os.path.join(file_path, shhs_dataset)
    
    df = pd.read_csv(file_path)
    selected_features = ['nsrrid','HvSnrd02', 'HOSnr02', 'LoudSn02','StpBrt02','SystBP', 'bmi_s1', 'gender','ESS_s1','DiasBP', 'NECK20', 'age_s1', 'race','HTNDerv_s1', 'STROKE15', 'smokstat_s1', 'ASTHMA15','Alcoh','Sleepy02','Drive02','ahi_a0h3a']
    df = df[selected_features]
    print("Total subjects = ", len(df))
    
    df = df[df['bmi_s1'].notna()]
    print("After remove BMI missing:", len(df))
    
    df = df[df['NECK20'].notna()]
    print("After remove Neck girth missing:", len(df))
    
    df = df[df['Alcoh'].notna()]
    print("After remove Alcoh", len(df))
    
    df = df[(df['HvSnrd02'].notna()) & (df['HvSnrd02'] != 8)]
    print("After remove Have you ever snored", len(df))
    
    df = df[~((df["LoudSn02"] == 8) & (df["HOSnr02"] != 8))]
    df = df[~((df["LoudSn02"].isna()) & (0 < df["HOSnr02"]) & (df["HOSnr02"] < 8))]
    print("After remove missing snoring loudness but indicating snoring freq", len(df))
    
    for i in range(5, len(selected_features)):
        pre_len = len(df)
        if selected_features[i] != 'Drive02':
            df = df[df[selected_features[i]].notna()]
            if len(df) != pre_len:
                print("After remove missing %s, %d" %(selected_features[i], len(df)))
            
    df['StpBrt02'] = df['StpBrt02'].fillna(0)
    df = df.reset_index(drop = True)
    df.to_csv("cleaned_shhs.csv")
    return df

def wsc_preprocessing(file_path):
    wsc_dataset = "wsc-dataset-0.1.0.csv"
    file_path = os.path.join(file_path, wsc_dataset)
    df = pd.read_csv(file_path)
    selected_features = ['wsc_id','wsc_vst', 'snore_freq', 'snore_vol','apnea_freq','hypertension_ynd','sitsysm', 'bmi', 'sex','ess','sitdiam', 'neck_girth1', 'age', 'race','alcohol_wk', 'ps_eds','ep8','remahi','nremahi']
    df = df[selected_features]
    print("Total subjects = ", len(df))
    
    df = df[df['wsc_vst'] == 1]
    print("After remove not from visit 1:", len(df))
        
    for i in range(2, len(selected_features)):
        pre_len = len(df)
        if selected_features[i] != 'ep8':
            df = df[df[selected_features[i]].notna()]
            if len(df) != pre_len:
                print("After remove missing %s, %d" %(selected_features[i], len(df)))
    
    #add REM and Non-rem AHI as the ground_truth
    ahi = []
    ahi_nonREM = list(df['nremahi'])
    ahi_REM = list(df['remahi'])
    for i in range(len(ahi_nonREM)):
        ahi.append(ahi_nonREM[i] + ahi_REM[i])
    df["ahi"] = ahi
    
    #Relable the variabels to be consistant with SHHS 1
    df['sex'] = df["sex"].replace("M", 1)
    df['sex'] = df["sex"].replace("F", 0)
    
    df['hypertension_ynd'] = df['hypertension_ynd'].replace("Y", 1)
    df['hypertension_ynd'] = df['hypertension_ynd'].replace("N", 0)
    
    df["snore_freq"] = df["snore_freq"].replace(9, 8)
    df["snore_freq"] = df["snore_freq"].replace(1, 0)
    df["snore_freq"] = df["snore_freq"].replace(2, 1)
    df["snore_freq"] = df["snore_freq"].replace(3, 2)
    df["snore_freq"] = df["snore_freq"].replace(4, 3)
    df["snore_freq"] = df["snore_freq"].replace(5, 4)
    
    df["snore_vol"] = df["snore_vol"].replace(9, 8)
    
    df["ps_eds"] = df["ps_eds"].replace(4, 5)
    df["ps_eds"] = df["ps_eds"].replace(3, 4)
    df["ps_eds"] = df["ps_eds"].replace(2, 3)
    df["ps_eds"] = df["ps_eds"].replace(1, 2)
    df["ps_eds"] = df["ps_eds"].replace(0, 1)
    
    df["apnea_freq"] = df["apnea_freq"].replace(9, 8)
    df["apnea_freq"] = df["apnea_freq"].replace(1, 0)
    df["apnea_freq"] = df["apnea_freq"].replace(2, 1)
    df["apnea_freq"] = df["apnea_freq"].replace(3, 1)
    df["apnea_freq"] = df["apnea_freq"].replace(3, 1)
    
    df["ep8"] = df["ep8"].replace(4, 0)
    df["ep8"] = df["ep8"].replace(3, 4)
    df["ep8"] = df["ep8"].replace(2, 3)
    df["ep8"] = df["ep8"].replace(1, 2)
    df["ep8"] = df["ep8"].replace(0, 1)
    
    
    #rename variables as same as in SHHS 1 for convenience
    df = df.rename(columns = {'snore_vol':'LoudSn02', 
                         'ps_eds':'Sleepy02',
                         'snore_freq':'HOSnr02',
                         'apnea_freq':'StpBrt02',
                         'hypertension_ynd':'HTNDerv_s1',
                         'bmi':'bmi_s1',
                         'age':'age_s1',
                         'sex':'gender',
                         'neck_girth1':'NECK20',
                         'sitsysm':'SystBP',
                         'sitdiam':'DiasBP',
                         'ess':'ESS_s1',
                         'alcohol_wk':'Alcoh',
                         'ep8':'Drive02',
                         'ahi':'ahi_a0h3a',
                         'wsc_id':'nsrrid'
                         })
    
    df = df.reset_index(drop = True)
    df.to_csv("wsc_testing_new.csv")
    return df

if __name__ == "__main__":
    file_path = os.path.dirname(__file__)
    _ =  shhs_preprocessing(file_path)
    _ = wsc_preprocessing(file_path)
    
    

