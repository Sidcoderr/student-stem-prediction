# Student STEM Inclination Prediction

## Overview
A hybrid fusion model combining a Learner Activity Model (LightGBM) with 
LLM-generated sentence embeddings (MiniLM) to predict whether a student 
will pursue STEM courses based on behavioral data from an educational platform.

## Results
- LAM Model Accuracy: 78.7%
- ROC-AUC: 0.876
- Dataset: 942,000+ student interaction records
- Features: 82 behavioral parameters
- Zero student overlap between train and validation sets confirmed

## Approach
- LightGBM for behavioral feature modeling (Learner Activity Model)
- MiniLM sentence transformer for LLM embeddings
- Fusion of LAM output + LLM embeddings for final STEM prediction
- Feature selection via correlation heatmap analysis
- Class balancing to handle imbalanced STEM vs non-STEM distribution

## Tech Stack
Python, LightGBM, Sentence Transformers, Pandas, NumPy, Scikit-learn
