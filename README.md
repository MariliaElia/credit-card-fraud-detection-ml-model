# Utilizing Generative Models to Address Imbalanced Data Classification in the Context of Credit Card Fraud Detection

Developed as part of my dissertation submitted to the University of Manchester for the degree of “M.Sc. Business Analytics: Operations research and Risk Analysis” in the Faculty of Humanities"

# Dissertation Overview

This study explored the effectiveness of data augmentation using generative models to address class imbalance in credit card datasets.

The two generative models tested are Generative Adversarial Network (GAN) and Variational Autoencoder (VAE), which are compared with traditional oversampling techniques, Synthetic Minority Oversampling Technique (SMOTE) and Adaptive Synthetic Sampling Approach for Imbalanced Learning (ADASYN). 


# Installation and Setup

## Codes and Resources Used
- **Editor Used:**  Visual Studio
- **Python Version:** Python 3.10.12

## Python Packages Used
- **General Purpose:** `copy, collections`
- **Data Manipulation:** `pandas, numpy` 
- **Data Visualization:** `seaborn, matplotlib`
- **Machine Learning:** `scikit-learn, tensorflow, keras`
- **Sampling:** `imblearn`

# Code structure
-	`visualisations.ipynb`: contains initial data exploration, including statistical summary table, correlation matrix, distribution graphs and boxplots.
-	`CV.py`: helper functions for implementing cross validation, and printing results.
-	`GAN.py`: GAN functions for training the model, generating synthetic samples, and concatenating with training data.
-	`VAE.py`: VAE functions for training the model, generating synthetic samples, and concatenating with training data.
-	`LR_model.ipynb, RF_model.ipynb, KNN_model.ipynb, XGB_model.ipynb`: training and evaluation of LR, RF, KNN, XGB respectively, with original distribution of data, SMOTE, ADASYN, VAE, and GAN


# Data Source
The dataset used is sourced from [Machine Learning Group - ULB](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and contains 284,807 credit card transactions made by European cardholders across two days in September 2013.

