## CV.py
# Includes helper functions for cross-validation and evaluation of models

# Performance metrics libraries
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score

# Visual Libraries
from matplotlib import pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

#Sampling Libraries
from imblearn.over_sampling import SMOTE, ADASYN

# Pre Libraries
import numpy as np
import pandas as pd 
from collections import Counter
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline

# Model Libraries
from sklearn.model_selection import StratifiedKFold

## Cross validation functions 
def visualize_confusion_matrix(cf_matrix):
    sns.heatmap(pd.DataFrame(cf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

def model_scores(y_true, y_pred):
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1 score:", f1_score(y_true, y_pred))
    print("ROC_AUC:", roc_auc_score(y_true, y_pred))

def cross_val_scores(scores):
    print("---------- Validation Results ---------------")
    print(f"Accuracy:, {np.mean(scores['test_accuracy']):0.6f} (+/- {np.std(scores['test_accuracy']):0.6f})")
    print(f"Precision: {np.mean(scores['test_precision']):0.6f} (+/- {np.std(scores['test_precision']):0.6f})")
    print(f"Recall: {np.mean(scores['test_recall']):0.6f} (+/- {np.std(scores['test_recall']):0.6f})")
    print(f"F1 score: {np.mean(scores['test_f1']):0.6f} (+/- {np.std(scores['test_f1']):0.6f})")
    print(f"ROC_AUC: {np.mean(scores['test_roc_auc']):0.6f} (+/- {np.std(scores['test_roc_auc']):0.6f})")

    print("---------- Training Results -------------")
    print(f"Accuracy:, {np.mean(scores['train_accuracy']):0.6f} (+/- {np.std(scores['train_accuracy']):0.6f})")
    print(f"Precision: {np.mean(scores['train_precision']):0.6f} (+/- {np.std(scores['train_precision']):0.6f})")
    print(f"Recall: {np.mean(scores['train_recall']):0.6f} (+/- {np.std(scores['train_recall']):0.6f})")
    print(f"F1 score: {np.mean(scores['train_f1']):0.6f} (+/- {np.std(scores['train_f1']):0.6f})")
    print(f"ROC_AUC: {np.mean(scores['train_roc_auc']):0.6f} (+/- {np.std(scores['train_roc_auc']):0.6f})")

def evaluate_model(X_test, y_test, estimator):
    num_fraud_cases_in_test = len(y_test[y_test==1])
    num_normal_cases_in_test = len(y_test[y_test==0])

    predictions = estimator.predict(X_test)
    cm = confusion_matrix(y_test, predictions)

    visualize_confusion_matrix(cm)

    # Print summary    
    print(f"\nClassified \t{cm[1,1]} out of {num_fraud_cases_in_test} \tfraud cases correctly")
    print(f"Misclassified \t{cm[0,1]} out of {num_normal_cases_in_test} normal cases")

    model_scores(y_test, predictions)

def model_cv(X_train, y_train, model, sampling_technique=None):
    skf = StratifiedKFold(n_splits=5)
    scores = {
        'train_accuracy': [],
        'test_accuracy': [],
        'train_precision': [],
        'test_precision': [],
        'train_recall': [],
        'test_recall': [],
        'train_f1': [],
        'test_f1': [],
        'train_roc_auc': [],
        'test_roc_auc': []
    }

    for train_index, test_index in skf.split(X_train, y_train):
        X_fold_train, X_fold_test = X_train.iloc[train_index], X_train.iloc[test_index]
        y_fold_train, y_fold_test = y_train.iloc[train_index], y_train.iloc[test_index]

        if sampling_technique == 'smote':
            sampler = SMOTE(random_state=1)
            X_fold_train_resampled, y_fold_train_resampled = sampler.fit_resample(X_fold_train, y_fold_train)
        elif sampling_technique == 'adasyn':
            sampler = ADASYN(random_state=1)
            X_fold_train_resampled, y_fold_train_resampled = sampler.fit_resample(X_fold_train, y_fold_train)
        else:
            X_fold_train_resampled, y_fold_train_resampled = X_fold_train, y_fold_train

        pipeline = Pipeline(steps=[('pre_process', StandardScaler()), ('classifier', model)])

        pipeline.fit(X_fold_train_resampled, y_fold_train_resampled)

        y_fold_train_pred = pipeline.predict(X_fold_train_resampled)
        y_fold_test_pred = pipeline.predict(X_fold_test)

        scores['train_accuracy'].append(accuracy_score(y_fold_train_resampled, y_fold_train_pred))
        scores['test_accuracy'].append(accuracy_score(y_fold_test, y_fold_test_pred))
        scores['train_precision'].append(precision_score(y_fold_train_resampled, y_fold_train_pred))
        scores['test_precision'].append(precision_score(y_fold_test, y_fold_test_pred))
        scores['train_recall'].append(recall_score(y_fold_train_resampled, y_fold_train_pred))
        scores['test_recall'].append(recall_score(y_fold_test, y_fold_test_pred))
        scores['train_f1'].append(f1_score(y_fold_train_resampled, y_fold_train_pred))
        scores['test_f1'].append(f1_score(y_fold_test, y_fold_test_pred))
        scores['train_roc_auc'].append(roc_auc_score(y_fold_train_resampled, y_fold_train_pred))
        scores['test_roc_auc'].append(roc_auc_score(y_fold_test, y_fold_test_pred))

    cross_val_scores(scores)
