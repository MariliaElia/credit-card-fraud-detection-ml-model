# Performance metrics libraries
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_validate

# Visual Libraries
from matplotlib import pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

#Sampling Libraries
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler

# Imported Libraries
import numpy as np
import pandas as pd 
import copy
from collections import Counter
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline

# Model Libraries
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold


def visualize_confusion_matrix(cf_matrix, axis):
    sns.heatmap(pd.DataFrame(cf_matrix), ax=axis, annot=True, cmap="YlGnBu" ,fmt='g')
    axis.set_title('Confusion matrix', y=1.1)
    axis.set_ylabel('Actual label')
    axis.set_xlabel('Predicted label')

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

def plot_roc_curve(model, y_true, X, axis):
    y_pred_proba = model.predict_proba(X)[::,1]

    fpr, tpr, _ = metrics.roc_curve(y_true,  y_pred_proba)
    auc = metrics.roc_auc_score(y_true, y_pred_proba)

    #create ROC curve
    axis.plot(fpr,tpr)
    axis.set_ylabel('Sensitivity/TPR/Recall')
    axis.set_xlabel('Specificity/FPR')
    axis.set_title("ROC-AUC="+str(auc))
    #plt.legend(loc=4)

def evaluate_model(X_test, y_test, estimator):
    num_fraud_cases_in_test = len(y_test[y_test==1])
    num_normal_cases_in_test = len(y_test[y_test==0])

    predictions = estimator.predict(X_test)
    cm = confusion_matrix(y_test, predictions)

    # Plot normalized confusion matrix and precision recall curve
    fig, axes = plt.subplots(1,2, figsize=(10,4))

    visualize_confusion_matrix(cm, axes[0])

    plot_roc_curve(estimator, y_test, X_test, axes[1])
    plt.tight_layout()

    # Print summary    
    print(f"\nClassified \t{cm[1,1]} out of {num_fraud_cases_in_test} \tfraud cases correctly")
    print(f"Misclassified \t{cm[0,1]} out of {num_normal_cases_in_test} normal cases")

    model_scores(y_test, predictions)

def model_cv(X_train, y_train, model, score, param_grid, sampling_technique = None):
    skf = StratifiedKFold(n_splits=5)

    if sampling_technique == 'smote':
        print("SMOTE applied")
        pipeline = Pipeline(steps = 
                            [['pre_process', StandardScaler()],
                            ['smote', SMOTE(random_state=1)],
                            ['classifier', model]])
    elif sampling_technique =='adasyn':
        print("ADASYN applied")
        pipeline = Pipeline(steps=
                            [['pre_process', StandardScaler()],
                            ['adasyn', ADASYN(random_state=1)],
                            ['classifier', model]])
    elif sampling_technique =='undersampling':
        pipeline = Pipeline(steps=
                            [['pre_process', StandardScaler()],
                            ['undersampling', RandomUnderSampler()],
                            ['classifier', model]])
    elif sampling_technique == 'original':
        print("Original applied")
        pipeline = Pipeline(steps = 
                            [['classifier', model]])
    else:
        pipeline = Pipeline(steps = 
                            [['pre_process', StandardScaler()],
                            ['classifier', model]])
    
    scores = cross_validate(pipeline, X_train, y_train, scoring=['accuracy','precision','recall','f1','roc_auc'],cv=skf, return_train_score=True)
    
    cross_val_scores(scores)
    
    # grid_search = GridSearchCV(estimator=pipeline,
    #                         param_grid=param_grid,
    #                         scoring = ['accuracy','precision','recall','f1','roc_auc'],
    #                         refit=score,
    #                         cv=skf,
    #                         n_jobs=-1)


    # grid_search.fit(X_train, y_train)

    # cv_results = pd.DataFrame(grid_search.cv_results_)
    # best_model_results = cv_results.loc[grid_search.best_index_]

    # print(f"Accuracy:, {best_model_results['mean_test_accuracy']:0.6f} (+/- {best_model_results['std_test_accuracy']:0.6f})")
    # print(f"Precision: {best_model_results['mean_test_precision']:0.6f} (+/- {best_model_results['std_test_precision']:0.6f})")
    # print(f"Recall: {best_model_results['mean_test_recall']:0.6f} (+/- {best_model_results['std_test_recall']:0.6f})")
    # print(f"F1 score: {best_model_results['mean_test_f1']:0.6f} (+/- {best_model_results['std_test_f1']:0.6f})")
    # print(f"ROC_AUC: {best_model_results['mean_test_roc_auc']:0.6f} (+/- {best_model_results['std_test_roc_auc']:0.6f})")

    # print('Best hyperparameters: ', grid_search.best_params_)

    return model