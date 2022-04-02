# OSA-Questionnaire

The questionnaire is used for obstructive sleep apnea risk prediction by using 6 questions: BMI, gender, hypertension history, age, neck girth, and snoring loudness. The quesionnaire will pre-classify subjects to two groups, followed by two independent LR model for prediction for each group. 
![image](https://user-images.githubusercontent.com/65090100/161368779-66370d44-8bef-4f25-8619-773e0a66d08f.png)


data_preprocess.py will preprocess the raw data of SHHS and WSC dataset from NSRR website:https://sleepdata.org/datasets, and generate two cleaned file for further analysis.

SHHS_feature_selection.py will perform the feature selection process based on normalized MI.

Statistic analysis.py will compare the characteristics of two testing datasets using t-test/chi-2 test depending on the variable types.

dual_LR_model_2022.py will mainly predict the OSA risk based on 6-item questions, and compare the performance with 95% CI with other widely accepted questionnaires: STOP-Bang, Berlin, ESS, 4-Variables


Please cite: Huo J, Quan S, Roveda J, et al. BASH-GN: A new machine learning derived questionnaire for screening obstructive sleep apnea[J]. medRxiv, 2022.
