import numpy as np
import pandas as pd

# Set variables
IDIR = 'INSPIRE/whole'
OUTCOME_VAR = 'aki'
INPUT_VARS = ['age', 'sex', 'emop', 'bmi', 'andur', 'htn', 'dm',
              'preop_hb', 'preop_platelet', 'preop_wbc', 'preop_aptt', 'preop_ptinr', 
              'preop_glucose', 'preop_bun', 'preop_albumin', 'preop_ast', 
              'preop_alt', 'preop_cr', 'preop_sodium', 'preop_potassium']

# important variables only
INPUT_VARS = ['age', 'sex', 'andur', 'preop_hb', 'preop_platelet', 'preop_wbc', 'preop_ptinr', 'preop_albumin', 'preop_cr', 'uro']

# Load operations
df = pd.read_csv(f'{IDIR}/operations.csv')

# find the first operation for each patient
df.sort_values('orin_time', inplace=True)
df = df.loc[df[['op_id','subject_id']].groupby('subject_id')['op_id'].idxmin()]

# Convert sex to binary
df['sex'] = df['sex'] == 'M'

# exclude cpb
df = df[df['cpbon_time'].isna() & df['cpboff_time'].isna()]

# exclude operations shorter than 2 hours
df.loc[:, 'andur'] = df['orout_time'] - df['orin_time']
# df = df[(df['andur'] > 2 * 60)]

# exclude operations other than general anesthesia
df = df[(df['antype'] == 'General')]

# exclude patients younger than 18 or older than 89
df = df[(df['age'] >= 18) & (df['age'] <= 89)]

# exclude patients with ASA class 6
df = df[(df['asa'] < 6)]

# one hot encoding for string variables
df['uro'] = df['department'] == 'UR'

# calculate BMI
valid_mask = df['height'] > 100
df['bmi'] = np.nan
df.loc[valid_mask, 'bmi'] = df.loc[valid_mask, 'weight'] / (df.loc[valid_mask, 'height'] / 100) ** 2

# Load labs
df_lab = pd.read_csv(f'{IDIR}/labs.csv')
df_lab.loc[df_lab['item_name'] == 'creatinine', 'item_name'] = 'cr'

# postop cr level within 7 days after surgery
item_name = 'cr'
df_crs = df_lab.loc[df_lab.item_name == 'cr'].copy()
df_crs = pd.merge(df, df_crs, on='subject_id', how='left')
df_crs = df_crs.loc[(df_crs.chart_time > df_crs.orout_time) & (df_crs.chart_time <= df_crs.orout_time + 7 * 3600 * 24)]
df_crs = df_crs.groupby('subject_id')['value'].max().reset_index()
df_crs.rename(columns={'value':'postop_cr'}, inplace=True)
df = pd.merge(df, df_crs, on='subject_id', how='left')

df_lab.sort_values('chart_time', inplace=True)
for item_name in ('hb', 'platelet', 'aptt', 'wbc', 'ptinr', 'glucose', 'bun', 'albumin', 'ast', 'alt', 'cr', 'sodium', 'potassium'):
    df = pd.merge_asof(df.sort_values('orin_time'), df_lab.loc[df_lab['item_name'] == item_name],
                    left_on='orin_time', right_on='chart_time', by='subject_id', tolerance=6* 30 * 24 * 60, suffixes=('', '_'))
    df.drop(columns=['chart_time', 'item_name'], inplace=True)
    df.rename(columns={'value':f'preop_{item_name}'}, inplace=True)

# exclude patients with preop cr > 3 mg/dL
df = df[(df['preop_cr'] <= 3) & (df['preop_cr'] > 0)]

# Load diagnoses
df_dx = pd.read_csv(f'{IDIR}/diagnosis.csv')
df_dx.sort_values('chart_time', inplace=True)
df_dx['value'] = 1

# Merge diagnosis data with operations to get orin_time for each diagnosis
df = pd.merge_asof(df.sort_values('orin_time'), df_dx.loc[df_dx['icd10_cm'].isin(('I10','I11','I12','I13','I15'))],
                left_on='orin_time', right_on='chart_time', by='subject_id', tolerance=6* 30 * 24 * 60, suffixes=('', '_'))
df.drop(columns=['chart_time', 'icd10_cm'], inplace=True)
df.rename(columns={'value':f'htn'}, inplace=True)

df = pd.merge_asof(df.sort_values('orin_time'), df_dx.loc[df_dx['icd10_cm'].isin(('E10','E11','E12','E13','E14'))],
                left_on='orin_time', right_on='chart_time', by='subject_id', tolerance=6* 30 * 24 * 60, suffixes=('', '_'))
df.drop(columns=['chart_time', 'icd10_cm'], inplace=True)
df.rename(columns={'value':f'dm'}, inplace=True)

# keep rows with non-missing preop and postop cr
df.dropna(subset=['preop_cr', 'postop_cr'], inplace=True)

# define AKI (KDIGO stage I or higher)
df[OUTCOME_VAR] = (df['postop_cr'] >= 1.5 * df['preop_cr']) | (df['postop_cr'] - df['preop_cr'] >= 0.3)

# Split a dataset into train and test sets
df = df.sample(frac=1, random_state=1).reset_index(drop=True)
ntrain = int(len(df) * 0.7)
y_train = df.loc[:ntrain, OUTCOME_VAR]
x_train = df.loc[:ntrain, INPUT_VARS].astype(float)
y_test = df.loc[ntrain:, OUTCOME_VAR]
x_test = df.loc[ntrain:, INPUT_VARS].astype(float)

# Print the number of train and test sets
print(f'{sum(y_train)}/{len(y_train)} ({np.mean(y_train)*100:.2f}%) train, {sum(y_test)}/{len(y_test)} ({np.mean(y_test)*100:.2f}%) test, {x_train.shape[1]} features', flush=True)

import xgboost as xgb
import shap
from sklearn.model_selection import RandomizedSearchCV
gs = RandomizedSearchCV(n_iter=100, estimator=xgb.sklearn.XGBClassifier(objective='binary:logistic'), n_jobs=-1, verbose=0,
                        param_distributions={
                            'learning_rate': [ 0.07, 0.01, 0.03, 0.05],
                            'max_depth': [3,4,5],
                            'n_estimators': [50, 75, 100],
                            'subsample': [0.5, 0.8, 1],
                            'colsample_bytree': [0.8, 1],
                        }, scoring='f1', cv=5)
gs.fit(x_train, y_train)
model = gs.best_estimator_.get_booster()

# Feature importance
import matplotlib.pyplot as plt
explainerXGB = shap.TreeExplainer(model)
shap_values_XGB_test = explainerXGB.shap_values(x_test)
plt.figure(figsize=(20, 5))
shap.summary_plot(shap_values_XGB_test, x_test, feature_names=INPUT_VARS, show=False)
plt.savefig('shap.png', bbox_inches="tight", pad_inches=1)
plt.close('all')

# Compute Metrics
import pickle
import scipy.stats
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, roc_auc_score, f1_score, precision_recall_curve, auc

pickle.dump(model, open('model.xgb', 'wb'))
y_pred = gs.predict_proba(x_test)[:, 1].flatten()

def roc_ci(y_true, y_pred):
    total = len(y_true)
    success = roc_auc_score(y_true, y_pred) * total
    alpha = 0.05
    lower = scipy.stats.beta.ppf(alpha / 2, success, total - success + 1)
    upper = scipy.stats.beta.ppf(1 - alpha / 2, success + 1, total - success)
    return lower, upper

fpr, tpr, thvals = roc_curve(y_test, y_pred)
lower_bound, upper_bound = roc_ci(y_test, y_pred)
auroc = auc(fpr, tpr)
precision, recall, _ = precision_recall_curve(y_test, y_pred)
auprc = auc(recall, precision)
optimal_idx = np.argmax(tpr - fpr)
thval = thvals[optimal_idx]
f1 = f1_score(y_test, y_pred > thval)
acc = accuracy_score(y_test, y_pred > thval)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred > thval).ravel()
print('auroc: {:.3f} ({:.3f} - {:.3f}), auprc: {:.3f}\tacc: {:.3f}\tf1: {:.3f}\tTN {}\tfp {}\tfn {}\tTP {}'.format(auroc, lower_bound, upper_bound, auprc, acc, f1, tn, fp, fn, tp))

plt.figure(figsize=(5,5))
plt.plot(fpr, tpr, label='GBM = {:0.3f}'.format(auroc))
plt.plot([0, 1], [0, 1], lw=1, linestyle='--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.savefig('auroc.png')
plt.close('all')