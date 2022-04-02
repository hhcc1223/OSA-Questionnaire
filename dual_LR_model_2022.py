# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 00:47:45 2021

@author: gudu0
"""

#import packages
import pandas as pd
import numpy as np
import math
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from sklearn import svm
from statistics import mean, stdev
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import os
import warnings

from data_preprocess import shhs_preprocessing, wsc_preprocessing
from SHHS_feature_selection import MI_select

warnings.filterwarnings('ignore')
np.random.seed(10)

class model:
    def __init__(self,df):       
        
        #Store the group info: low phenotype: 0, and high phenotype: 1
        self.group = []
        
        #FourVariable result
        self.four = []
        self.FourVariables(df)
        
        #StopBang result
        self.score = []
        self.stopbang = []
        self.STOP_Bang(df)
        
        #ESS result
        self.ess = []
        self.ESS(df)
        
        #Berlin result
        self.berlin = []
        self.Berlin(df)
        
        #selected features
        features = ['nsrrid','HTNDerv_s1','bmi_s1','age_s1','NECK20','gender','LoudSn02']

        self.df = df[features]
        self.df['gender'] = self.df['gender'].replace(2, 0)
        #if no data, simply means not snoring at all, filled with 0
        self.df['LoudSn02'] = self.df['LoudSn02'].fillna(0)
        #combine don't know with not snoring
        self.df['LoudSn02'] = self.df['LoudSn02'].replace(8,0)
        
        
        self.df['LoudSn02'] = self.df['LoudSn02'].replace(0,5)
        self.df['LoudSn02'] = self.df['LoudSn02'].replace(1,0)
        self.df['LoudSn02'] = self.df['LoudSn02'].replace(2,1)
        self.df['LoudSn02'] = self.df['LoudSn02'].replace(3,2)
        self.df['LoudSn02'] = self.df['LoudSn02'].replace(4,3)
        self.df['LoudSn02'] = self.df['LoudSn02'].replace(5,4)
        
        #one hot encoding the LoudSn02, and drop the original column
        LoudSn02 = self.bne(self.df.LoudSn02)
        self.df= pd.concat([self.df,pd.DataFrame(LoudSn02)],axis = 1)
        self.df = self.df.rename(columns = {0:'LoudSn02_1',1:'LoudSn02_2',2:'LoudSn02_3'})
        self.df = self.df.drop(columns = ['LoudSn02'])
        
        #add group column based on risk factor calssification
        self.df['group'] = self.group

        #convert to numpy for futher usage
        self.group = np.array(self.group)
        
        #add other questionnaire score and result to dataframe
        self.df['score_sb'] = self.stopbang
        self.df['ESS'] = self.ess
        self.df['fourvariable'] = self.four
        self.df['berlin'] = self.berlin
        self.df['output'] = df['output']
        
        # split to testing and training
        Y = self.df['output']       
        self.array = self.df.values
        
        #first col is id
        X = self.array[:,1:14]
        Y = self.array[:,14]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                    X,Y,test_size = 0.3,random_state = 10, shuffle = False)
        
        
        #low / high phenotype
        self.group_train = self.X_train[:,8]
        self.group_test = self.X_test[:,8]
        #stopbang score
        self.score_sb = self.X_test[:,9]
        #ess score
        self.score_ess = self.X_test[:,10]
        #Four vairable score
        self.score_four = self.X_test[:,11]
        #Berlin result
        self.score_berlin = self.X_test[:,12]
        
        self.X_train = self.X_train[:,0:8]
        self.X_test = self.X_test[:,0:8]
        
        self.cross_validation()
        
        #show predict result   
    def show_result(self,algo,y_pred, y_pred_proba):
        cm = confusion_matrix(self.y_test,y_pred)
        print(algo)
        print(cm)
        print(classification_report(self.y_test, y_pred))
        print('ROC_AUC_socre=', roc_auc_score(self.y_test,y_pred_proba,multi_class = 'ovr'))
        print('AUPRC_socre=', average_precision_score(self.y_test,y_pred_proba))
        self.CI_Cal(self.y_test, y_pred_proba)
    
    #calculate credible intervals using bootstrapping    
    def CI_Cal(self, y_true, y_pred_proba):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred_proba)
        threshold = 0.427471
        y_pred1 = np.where(y_pred < threshold, 0, y_pred)
        y_pred1 = np.where(y_pred1 >= threshold, 1, y_pred1)
        print("f1_score = ", f1_score(y_true, y_pred1))
        
        n_bootstraps = 1000
        rng_seed = 42
        auroc_scores = []
        f1_scores = []
        auprc_scores = []
        rng = np.random.RandomState(rng_seed)
        for i in range(n_bootstraps):
            indices = rng.randint(0, len(y_pred), len(y_pred))
            if len(np.unique(y_true[indices])) < 2:
                # We need at least one positive and one negative sample for ROC AUC
                # to be defined: reject the sample
                continue
        

            auroc_score = roc_auc_score(y_true[indices], y_pred[indices])
            auroc_scores.append(auroc_score)
            f1 = f1_score(y_true[indices], y_pred1[indices])
            f1_scores.append(f1)
            auprc_score = average_precision_score(y_true[indices], y_pred[indices])
            auprc_scores.append(auprc_score)
            
        
        sorted_auroc_scores = np.array(auroc_scores)
        sorted_auroc_scores.sort()
        confidence_lower_auroc = sorted_auroc_scores[int(0.05 * len(sorted_auroc_scores))]
        confidence_upper_auroc = sorted_auroc_scores[int(0.95 * len(sorted_auroc_scores))]
        
        sorted_f1_scores = np.array(f1_scores)
        sorted_f1_scores.sort()
        confidence_lower_f1 = sorted_f1_scores[int(0.05 * len(sorted_f1_scores))]
        confidence_upper_f1 = sorted_f1_scores[int(0.95 * len(sorted_f1_scores))]
        
        sorted_auprc_scores = np.array(auprc_scores)
        sorted_auprc_scores.sort()
        confidence_lower_auprc = sorted_auprc_scores[int(0.05 * len(sorted_auprc_scores))]
        confidence_upper_auprc = sorted_auprc_scores[int(0.95 * len(sorted_auprc_scores))]
        
        print("Credible interval for the auroc_score: [{:0.3f} - {:0.3}]".format(confidence_lower_auroc, confidence_upper_auroc))
        print("Credible interval for the f1_score: [{:0.3f} - {:0.3}]".format(confidence_lower_f1, confidence_upper_f1))
        print("Credible interval for the auprc_score: [{:0.3f} - {:0.3}]".format(confidence_lower_auprc, confidence_upper_auprc))
        
    
    #binary encoder
    def bne(self, feature_array):
        unique = np.unique(feature_array)
        m = math.ceil(math.log2(len(unique)))
        dic = {}
        val = 0
        s = "{0:0" + str(m) + 'b}'
        for num in unique:
            if num not in dic.keys():
                dic[num] = s.format(val)
                val += 1
    
        transformed_array = np.empty([len(feature_array),m],dtype = int)
        for i in range(len(feature_array)):
            trans_num = dic[feature_array[i]]
            for j in range(len(trans_num)):
                transformed_array[i][j] = trans_num[j]
        return transformed_array
    
    #establish different classifiers
    def classification(self,clf,X_train,n=2):
        tree_dict = {}
        for i in range(n):
            if i == 0: 
                tree_dict[i] = LogisticRegression(random_state = 10)
            elif i == 1:
                tree_dict[i] = LogisticRegression(random_state = 10)
            else:
                tree_dict[i] = LogisticRegression(random_state = 10)
            x_i = X_train[self.group_train == i, :]
            y_i = self.y_train[self.group_train == i]
            tree_dict[i] = tree_dict[i].fit(x_i,y_i)
            self.print_coef(tree_dict[i])
        return tree_dict
        
    
    #plot ROC curve
    def ROC_curve(self,y_pred_proba):
        fpr, tpr, t= roc_curve(self.y_test, y_pred_proba[:,1])
        data = {"fpr": fpr, "tpr":tpr, "t":t}
        self.roc = pd.DataFrame(data = data)
        fpr_sb, tpr_sb, t_sb= roc_curve(self.y_test, self.score_sb)
        fpr_ess, tpr_ess, t_ess= roc_curve(self.y_test, self.score_ess)
        fpr_4, tpr_4, t_4= roc_curve(self.y_test, self.score_four)
        fpr_berlin, tpr_berlin, t_berlin= roc_curve(self.y_test, self.score_berlin)
        precision, recall, thr= precision_recall_curve(self.y_test, y_pred_proba[:,1])
        precision_sb, recall_sb, thr_sb= precision_recall_curve(self.y_test, self.score_sb)
        precision_ess, recall_ess, thr_ess= precision_recall_curve(self.y_test, self.score_ess)
        precision_4, recall_4, thr_4= precision_recall_curve(self.y_test, self.score_four)
        gmeans = np.sqrt(tpr*(1-fpr))
        idx = np.argmax(gmeans)
        gmeans_sb = np.sqrt(tpr_sb*(1-fpr_sb))
        idx_sb = np.argmax(gmeans_sb)
        gmeans_ess = np.sqrt(tpr_ess*(1-fpr_ess))
        idx_ess = np.argmax(gmeans_ess)
        gmeans_4 = np.sqrt(tpr_4*(1-fpr_4))
        idx_4 = np.argmax(gmeans_4)
        gmeans_berlin = np.sqrt(tpr_berlin*(1-fpr_berlin))
        idx_berlin = np.argmax(gmeans_berlin)
        print('Proposed algorithm:Best Threshold=%f, G-Mean=%.3f' % (t[idx], gmeans[idx]))
        print('Sensitivity = ',tpr[idx])
        print('Specificity = ',1-fpr[idx])
        print('')
        print('Stopbang: Best Threshold=%f, G-Mean=%.3f' % (t_sb[idx_sb], gmeans_sb[idx_sb]))
        cm = confusion_matrix(self.y_test, self.score_sb)
        # self.plot_cm(cm, "STOP-Bang")
        cm = confusion_matrix(self.y_test[self.group_test == 0],self.score_sb[self.group_test == 0])
        print("G0", cm)
        # self.plot_cm(cm, "STOP-Bang Low Phenotype Group")
        cm = confusion_matrix(self.y_test[self.group_test == 1],self.score_sb[self.group_test == 1])
        print("G1", cm)
        print('Sensitivity = ',tpr_sb[idx_sb])
        print('Specificity = ',1-fpr_sb[idx_sb])
        print('AUPRC_score = ', average_precision_score(self.y_test, self.score_sb))
        print('AUC:',roc_auc_score(self.y_test, self.score_sb))
        self.CI_Cal(self.y_test, self.score_sb)
        print('')
        print('ESS: Best Threshold=%f, G-Mean=%.3f' % (t_ess[idx_ess], gmeans_ess[idx_ess]))
        cm = confusion_matrix(self.y_test, self.score_ess)
        # self.plot_cm(cm, "ESS")
        cm = confusion_matrix(self.y_test[self.group_test == 0],self.score_ess[self.group_test == 0])
        print("G0", cm)
        # self.plot_cm(cm, "ESS Low Phenotype Group")
        cm = confusion_matrix(self.y_test[self.group_test == 1],self.score_ess[self.group_test == 1])
        print("G1", cm)
        print('Sensitivity = ',tpr_ess[idx_ess])
        print('Specificity = ',1-fpr_ess[idx_ess])
        print('AUPRC_score = ', average_precision_score(self.y_test, self.score_ess))
        print('AUC:',roc_auc_score(self.y_test, self.score_ess))
        self.CI_Cal(self.y_test, self.score_ess)
        print('')
        print('Four-variables: Best Threshold=%f, G-Mean=%.3f' % (t_4[idx_4], gmeans_4[idx_4]))
        cm = confusion_matrix(self.y_test, self.score_four)
        # self.plot_cm(cm, "Four-Variable")
        cm = confusion_matrix(self.y_test[self.group_test == 0],self.score_four[self.group_test == 0])
        print("G0", cm)
        # self.plot_cm(cm, "Four-Variables Low Phenotype Group")
        cm = confusion_matrix(self.y_test[self.group_test == 1],self.score_four[self.group_test == 1])
        print("G1", cm)
        print('Sensitivity = ',tpr_4[idx_4])
        print('Specificity = ',1-fpr_4[idx_4])
        print('AUPRC_score = ', average_precision_score(self.y_test, self.score_four))
        print('AUC:',roc_auc_score(self.y_test, self.score_four))
        self.CI_Cal(self.y_test, self.score_four)
        print('')
        print('Berlin: Best Threshold=%f, G-Mean=%.3f' % (t_berlin[idx_4], gmeans_berlin[idx_berlin]))
        cm = confusion_matrix(self.y_test, self.score_berlin)
        # self.plot_cm(cm, "Berlin")
        cm = confusion_matrix(self.y_test[self.group_test == 0],self.score_berlin[self.group_test == 0])
        print("G0", cm)
        # self.plot_cm(cm, "Berlin Low Phenotype Group")
        cm = confusion_matrix(self.y_test[self.group_test == 1],self.score_berlin[self.group_test == 1])
        # self.plot_cm(cm, "Berlin High Phenotype Group")
        print("G1", cm)
        print('Sensitivity = ',tpr_berlin[idx_berlin])
        print('Specificity = ',1-fpr_berlin[idx_berlin])
        print('AUPRC_score = ', average_precision_score(self.y_test, self.score_berlin))
        print('AUC:',roc_auc_score(self.y_test, self.score_berlin))
        self.CI_Cal(self.y_test, self.score_berlin)
        
        sns.set(font_scale = 1.5, style = 'white')
        plt.figure(figsize = (8, 6), dpi = 300)
        plt.plot(fpr,tpr, label = 'BASH-GN')
        plt.plot(fpr_sb,tpr_sb,linestyle = '--', label = 'STOP-Bang')
        plt.plot(fpr_berlin,tpr_berlin,linestyle = 'dashdot', label = 'Berlin')
        plt.plot(fpr_4,tpr_4,linestyle = 'dashdot', label = 'Four-Variable')
        plt.plot(fpr_ess,tpr_ess,linestyle = 'dashed', label = 'ESS')
        plt.scatter(fpr[idx], tpr[idx], marker='o', color='black', label='Optimal')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.title('ROC curve')
        plt.legend()
        
        plt.show()
    
    #Perform test
    def test(self,tree_dict,X_test):
        
                
        y_pred, y_pred_proba = [], np.empty((0,2),float)
        y_pred1, y_pred_proba1 = [], np.empty((0,2),float)
        y_pred2, y_pred_proba2 = [], np.empty((0,2),float)
        
        for i in range(len(self.group_test)):
            tree_num = self.group_test[i]
            # y_pred_i = tree_dict[tree_num].predict(np.reshape(X_test[i],(-1,8)))
            y_pred_proba_temp = tree_dict[tree_num].predict_proba(np.reshape(X_test[i],(-1,8)))
            if self.group_test[i] == 0:
                if y_pred_proba_temp[:,1] < 0.42741:
                    y_pred_i = 0
                else:
                    y_pred_i = 1
            elif self.group_test[i] == 1:
                if y_pred_proba_temp[:,1] < 0.42741:
                    y_pred_i = 0
                else:
                    y_pred_i = 1
            y_pred_proba = np.append(y_pred_proba, y_pred_proba_temp, axis = 0)
            y_pred.append(y_pred_i)
                
            
            #test seperately based on group: low or high phenotype
            if self.group_test[i] == 0:
                y_pred1.append(y_pred_i)
                y_pred_proba1 = np.append(y_pred_proba1, y_pred_proba_temp, axis = 0)
            else:
                y_pred2.append(y_pred_i)
                y_pred_proba2 = np.append(y_pred_proba2, y_pred_proba_temp, axis = 0)
               
        return y_pred, y_pred_proba
    
    def plot_cm(self, confustion_matrix, title, normalized = True):
        if normalized:
            confustion_matrix = confustion_matrix.astype("float")/confustion_matrix.sum(axis = 1)[:, np.newaxis]
        plt.figure(figsize = (10,7))
        cm = pd.DataFrame(data = confustion_matrix, columns=["Low Risk", "High Risk"], index = ["Low Risk", "High Risk"])
        sns.set(font_scale = 1.5)
        sns.heatmap(cm, annot=True, fmt = ".2f", cmap = "Blues", vmin = 0.1, vmax = 1.00)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(title)
        plt.show()
        
    
    #cross validation    
    def cross_validation(self):
        
        AUROC_score = []
        
        #cross validation
        clf = {'LR':LogisticRegression(random_state = 10),
                "SVC":svm.SVC(probability = True,kernel = 'rbf'),
                "KNN":KNeighborsClassifier(n_neighbors = int(len(self.df)**0.5), p = 4, metric = 'euclidean'),
                "DT":DecisionTreeClassifier(random_state = 10),
                "ET":ExtraTreesClassifier(n_estimators = 30, random_state = 10),
                'Ada':AdaBoostClassifier(n_estimators = 100, random_state = 10),
                "GNB":GaussianNB(),
                'RF':RandomForestClassifier(n_estimators = 100, bootstrap = False,random_state = 10),
                      }
        
        for clf_number in range(2):
            for name in clf:
                clf_dt = clf[name]
                temp_score = []
                X = self.array[:,1:9][self.group == clf_number]
                Y = self.array[:,14][self.group == clf_number]
                from sklearn import preprocessing
                sc_X = preprocessing.MinMaxScaler() 
                skf = StratifiedKFold(n_splits =10, shuffle = True, random_state = 10)
                lst_accu_stratified = []
                
                for train_index, test_index in skf.split(X,Y):
                    x_train_fold, x_test_fold = X[train_index], X[test_index] 
                    y_train_fold, y_test_fold = Y[train_index], Y[test_index]
                    x_train_fold = sc_X.fit_transform(x_train_fold)
                    x_test_fold = sc_X.fit_transform(x_test_fold)
                    clf_dt.fit(x_train_fold, y_train_fold)
                    y_pred_proba = clf_dt.predict_proba(x_test_fold)
                    fpr, tpr, t= roc_curve(y_test_fold, y_pred_proba[:,1])
                    from sklearn import metrics
                    lst_accu_stratified.append(metrics.auc(fpr, tpr))
                
                temp_score.append(mean(lst_accu_stratified))
                temp_score.append(stdev(lst_accu_stratified))
                if clf_number == 0: temp_score.append("Low Phenotype")
                else: temp_score.append("High Phenotype")
                temp_score.append(name)
                
                AUROC_score.append(temp_score)
                
                
            
                # print('List of possible AUROC for ' + str(clf_number) + ':', lst_accu_stratified) 
                # print('\nMaximum AUROC That can be obtained from this model is:', 
                #       max(lst_accu_stratified)) 
                # print('\nMinimum AUROC:', 
                #       min(lst_accu_stratified)) 
                # print('\nOverall AUROC:', 
                #       mean(lst_accu_stratified)) 
                # print('\nStandard Deviation is:', stdev(lst_accu_stratified))
                
        plt.figure(figsize=(8, 6), dpi=300)
        df_auroc = pd.DataFrame(data = AUROC_score, columns = ["AUROC mean", "std", "Group", "Algorithms"])
        sns.set(font_scale=1.5, style='white')
        ax2 = sns.barplot(data = df_auroc, x = "Algorithms", y = "AUROC mean", hue = "Group", palette=("pastel"), hue_order=("Low Phenotype","High Phenotype"))
        ax2.set(ylim=(0.4, 0.9))
        ax2.set_title("AUROC mean of stratified 10-fold cross validation using different algorithms")
        for i in range(len(df_auroc)//2):
            plt.errorbar(i - 0.2, df_auroc.iloc[i]["AUROC mean"], yerr = df_auroc.iloc[i]["std"], capsize=3.0, ecolor="black", capthick=(1.5))
            plt.errorbar(i + 0.2, df_auroc.iloc[i + 8]["AUROC mean"], yerr = df_auroc.iloc[i+8]["std"], capsize=3.0, ecolor="black", capthick=(1.5))
    
    def print_coef(self, clf):
        
        #revert the standarlization for interpretion
        """
        after standarlization:
            coef * (x_standarlized) = coef * (x - u)/std
            -> coef * x/std - coef * u/std
            ->new_coef = coef/std, intercept -= coef * u/std
        """
        means = list(self.sc_X.mean_)
        var = list(self.sc_X.var_)
        original_coefs = list(clf.coef_[0])
        coef = []
        intercept = clf.intercept_[0]
        
        for i in range(len(original_coefs)):
            coef.append(original_coefs[i]/(var[i]) ** 0.5)
            intercept -= original_coefs[i] * means[i]/ (var[i]) ** 0.5
        
        
        print('coef:',coef)
        print('intercept:',intercept)

    #Logistic regression
    def LR(self):
        clf = LogisticRegression(random_state = 10)
        self.sc_X = StandardScaler()
        X_train = self.sc_X.fit_transform(self.X_train)
        X_test = self.sc_X.transform(self.X_test)
        self.tree_dict = self.classification(clf,X_train)
        
        y_pred, y_pred_proba = self.test(self.tree_dict,X_test)
        self.ROC_curve(y_pred_proba)

        #2-category
        self.show_result('Proposed dual model', y_pred, y_pred_proba[:,1])
        
    #ESS score
    def ESS(self,df):
        for i in range(len(df)):
            if df.ESS_s1.iloc[i] < 11:
                self.ess.append(0)
            else: self.ess.append(1)
    
    #Four Variable tool calculation
    def FourVariables(self, df):
        for i in range(len(df)):
            if df['bmi_s1'].iloc[i] < 21:
                bmi_cate = 1
            elif 21 <= df['bmi_s1'].iloc[i] < 23:
                bmi_cate = 2
            elif 23 <= df['bmi_s1'].iloc[i] < 25:
                bmi_cate = 3
            elif 25 <= df['bmi_s1'].iloc[i] < 27:
                bmi_cate = 4
            elif 27 <= df['bmi_s1'].iloc[i] < 30:
                bmi_cate = 5
            else:
                bmi_cate = 6
            if df['HOSnr02'].iloc[i] == 3 or df['HOSnr02'].iloc[i] == 4:
                snoring_cate = 1
            else:
                snoring_cate = 0
            if df['SystBP'].iloc[i]< 140 or df['DiasBP'].iloc[i] < 90:
                bp_cate = 1
            elif 140 <= df['SystBP'].iloc[i] < 160 or 90 <= df['DiasBP'].iloc[i] < 100:
                bp_cate = 2
            elif 160 <= df['SystBP'].iloc[i] < 180 or 100 <= df['DiasBP'].iloc[i] < 110:
                bp_cate = 3
            else:
                bp_cate = 4
            if df['gender'].iloc[i] * 4 + bmi_cate + bp_cate + snoring_cate*4 < 14:
                self.four.append(0)
            else:
                self.four.append(1)
    
    #use Berlin
    def Berlin(self, df):
        
        for i in range(len(df)):
            category1, category2, category3 = 0, 0, 0
            #do you snore + snoring loudness, as long as snoring loudness is indicated, the subject snores
            if 0 < df.LoudSn02.iloc[i] < 5:
                category1 += 1
                if 4 >= df.LoudSn02.iloc[i] >= 3:
                    category1 += 2
                if 4 >= df.HOSnr02.iloc[i] >= 3:
                    category1 += 1
            if int(df.StpBrt02.iloc[i]) == 1:
                category1 += 2
                
            if df.Sleepy02.iloc[i] > 4:
                category2 += 2
            
            if df.Drive02.iloc[i] >= 2: 
                category2 += 1
            
            if int(df.HTNDerv_s1.iloc[i]) == 1 or df.bmi_s1.iloc[i] > 30:
                category3 += 1
        
            if category1 >= 2: category1 = 1
            if category2 >= 2: category2 = 1
            if category1 + category2 + category3 >= 2: self.berlin.append(1)
            else: self.berlin.append(0)
            
    
    
    #use STOPBANG 
    def STOP_Bang(self,df):
        for i in range(len(df)):
            
            #to use the threshold(=3) to classify subjects as low/high phenotype
            score1 = 0
            
            #stop and score is to caculate the STOP score and stopband score for final STOPBANG classification
            stop = 0
            score = 0
            
            if 2 < df.LoudSn02.iloc[i] < 5:
                score += 1
                stop += 1
                score1 += 1
            if df.Sleepy02.iloc[i] > 3:
                score += 1
                stop += 1
            if int(df.StpBrt02.iloc[i]) == 1:
                score += 1
                stop += 1
            if int(df.HTNDerv_s1.iloc[i]) == 1:
                score += 1
                score1 += 1
                stop += 1
            if df.bmi_s1.iloc[i] > 35:
                score += 1
                score1 += 1
            if df.age_s1.iloc[i] > 50:
                score += 1
                score1 += 1
            if df.NECK20.iloc[i] > 40:
                score += 1
                score1 += 1
            if df.gender.iloc[i] == 1:
                score += 1
                score1 += 1
            
            if score1 <= 2: self.group.append(0)
            else: self.group.append(1) 
            
            #Based on the STOPBang definition
            if score >= 3:
                self.score.append(2)
            elif stop >= 2:
                if int(df.gender.iloc[i]) == 1 or df.bmi_s1.iloc[i] >= 35 or df.NECK20.iloc[i] >= 40:
                    self.score.append(2)
                else:
                    self.score.append(1)
            elif score <= 2:
                self.score.append(0)
            else:
                self.score.append(1)  
            if score >= 3:
                self.stopbang.append(1)
            else: self.stopbang.append(0)
    
    def website_check(self, data_dict):
        
        score, group = 0, 0
        if data_dict["HTNDerv_s1"] > 0: score += 1
        if data_dict["bmi_s1"] >= 35: score += 1
        if data_dict["age_s1"] > 50: score += 1
        if data_dict["NECK20"] > 40: score += 1
        if data_dict["gender"] > 0: score += 1
        if 1 < data_dict["LoudSn02"] < 4: score += 1

        if score >= 3: group = 1
        
        data_df = pd.DataFrame(data = data_dict, columns = ['HTNDerv_s1','bmi_s1', 'age_s1', 'NECK20', 'gender','LoudSn02'], index = [0])
        LoudSn02 = bin(data_dict["LoudSn02"]).replace("0b", "").zfill(3)
        LoudSn02 = [[int(char)] for char in LoudSn02]
        
        data_df= pd.concat([data_df,pd.DataFrame(LoudSn02).T],axis = 1)
        data_df = data_df.rename(columns = {0:'LoudSn02_1',1:'LoudSn02_2',2:'LoudSn02_3'})
        data_df = data_df.drop(columns = ['LoudSn02'])
        
        X = data_df.values
        X_test = self.sc_X.transform(X)
        y_pred_proba = self.tree_dict[group].predict_proba(np.reshape(X_test,(-1,8)))
        if group == 0:
            if y_pred_proba[:,1] < 0.427:
                y_pred = 0
            else:
                y_pred = 1
        elif group == 1:
            if y_pred_proba[:,1] < 0.427:
                y_pred = 0
            else:
                y_pred = 1
        return  group, y_pred_proba[:, 1], y_pred
    
    #for independent dataset testing
    def independent_testing(self, df):
        #low phenotype: 0, and high phenotype: 1
        self.group = []
        df['StpBrt02'] = df['StpBrt02'].fillna(0)
        #FourVariable result
        self.four = []
        self.FourVariables(df)
        
        #StopBang result
        self.score = []
        self.stopbang = []
        self.STOP_Bang(df)
        
        #ESS result
        self.ess = []
        self.ESS(df)
        
        #Berlin result
        self.berlin = []
        self.Berlin(df)
        
        #selected features
        features = ['nsrrid','HTNDerv_s1','bmi_s1','age_s1','NECK20','gender','LoudSn02']
        self.df = df[features]
        
        #if no data, simply means not snoring at all, filled with 0
        self.df['LoudSn02'] = self.df['LoudSn02'].fillna(0)
        #combine don't know with not snoring
        self.df['LoudSn02'] = self.df['LoudSn02'].replace(8,0)
        
        self.df['LoudSn02'] = self.df['LoudSn02'].replace(0,5)
        self.df['LoudSn02'] = self.df['LoudSn02'].replace(1,0)
        self.df['LoudSn02'] = self.df['LoudSn02'].replace(2,1)
        self.df['LoudSn02'] = self.df['LoudSn02'].replace(3,2)
        self.df['LoudSn02'] = self.df['LoudSn02'].replace(4,3)
        self.df['LoudSn02'] = self.df['LoudSn02'].replace(5,4)
        
        
        
        #one hot encoding the LoudSn02, and drop the original column
        LoudSn02 = self.bne(self.df.LoudSn02)
        self.df= pd.concat([self.df,pd.DataFrame(LoudSn02)],axis = 1)
        self.df = self.df.rename(columns = {0:'LoudSn02_1',1:'LoudSn02_2',2:'LoudSn02_3'})
        self.df = self.df.drop(columns = ['LoudSn02'])
        
        #add group column based on risk factor calssification
        self.df['group'] = self.group

        
        #convert to numpy for futher usage
        self.group = np.array(self.group)
        
        #add other questionnaire score and result to dataframe
        self.df['score_sb'] = self.stopbang
        self.df['ESS'] = self.ess
        self.df['fourvariable'] = self.four
        self.df["Berlin"] = self.berlin
        self.df['output'] = df['output']
        
        # split to testing and training    
        self.array = self.df.values
        
        #first col is id
        X = self.array[:,1:14]
        self.y_test = self.array[:,14]
        
        #low / high phenotype
        self.group_test = X[:,8]
        #stopbang score
        self.score_sb = X[:,9]
        #ess score
        self.score_ess = X[:,10]
        #Four vairable score
        self.score_four = X[:,11]
        self.score_berlin = X[:, 12]
        
        self.X_test = X[:,0:8]
        X_test = self.sc_X.transform(self.X_test)
        y_pred, y_pred_proba = self.test(self.tree_dict,X_test)
        self.ROC_curve(y_pred_proba)
        self.show_result('Proposed dual model', y_pred, y_pred_proba[:,1])

#Groud truth
def Predict_condition(df1, condition):
    output = []
    if condition == 'Severe':
        for i in range(len(df1.ahi_a0h3a)):
            if df1.ahi_a0h3a[i] < 5:
                output.append(0)
            elif 5 <= df1.ahi_a0h3a[i] < 15:
                output.append(0)
            elif 15 <= df1.ahi_a0h3a[i] < 30:
                output.append(0)
            else:
                output.append(1)
    elif condition == 'Moderate':
        for i in range(len(df1.ahi_a0h3a)):
            if df1.ahi_a0h3a[i] < 5:
                output.append(0)
            elif 5 <= df1.ahi_a0h3a[i] < 15:
                output.append(0)
            elif 15 <= df1.ahi_a0h3a[i] < 30:
                output.append(1)
            else:
                output.append(1)
    elif condition == 'Mild':
        for i in range(len(df1.ahi_a0h3a)):
            if df1.ahi_a0h3a[i] < 5:
                output.append(0)
            elif 5 <= df1.ahi_a0h3a[i] < 15:
                output.append(1)
            elif 15 <= df1.ahi_a0h3a[i] < 30:
                output.append(1)
            else:
                output.append(1)
    elif condition == 'ALL':
        for i in range(len(df1.ahi_a0h3a)):
            if df1.ahi_a0h3a[i] < 5:
                output.append(0)
            elif 5 <= df1.ahi_a0h3a[i] < 15:
                output.append(1)
            elif 15 <= df1.ahi_a0h3a[i] < 30:
                output.append(2)
            else:
                output.append(3)
    else: raise NameError('Input condition not supported, please type one of followings: Mild, Moderate, Severe, ALL')
    return output


if __name__ == "__main__":
    my_path = os.path.abspath(os.path.dirname(__file__))
    df_shhs = shhs_preprocessing(my_path)
    df_wsc = wsc_preprocessing(my_path)
    
    #Feature selection
    NMI = MI_select(my_path)
    
    features = ['nsrrid','HTNDerv_s1','LoudSn02','StpBrt02','Sleepy02','bmi_s1','age_s1','NECK20','gender','HOSnr02','SystBP','DiasBP','ESS_s1','Alcoh','Drive02','ahi_a0h3a']
    df_shhs = df_shhs[features]
    df_wsc = df_wsc[features]
    
    #Change ground Truth label to binary
    df_shhs['output'] = Predict_condition(df_shhs, 'Moderate')
    df_wsc['output'] = Predict_condition(df_wsc, 'Moderate')
    
    #Perform cross-validation for algorithm selection
    m = model(df_shhs)
    
    #Analysis
    m.LR()
    m.independent_testing(df_wsc)
    
    
    
    

