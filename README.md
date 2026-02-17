# 📊 Financial Risk Analysis — Credit Default Prediction

This project implements a production-style machine learning pipeline to predict borrower default risk using the Give Me Some Credit dataset (250,000 borrowers).

The objective is to estimate the probability that a borrower will experience serious financial distress within two years and translate predictions into actionable risk tiers.

## 🎯 Problem Context

Banks rely on credit scoring models to decide who receives financing.
Accurate risk prediction reduces loan losses while maintaining access to credit.

This project focuses not only on predictive performance but also on business-aligned decision thresholds in an imbalanced dataset.

## 🏗 Project Structure
Financial_Risk_Analysis/  

├── src/                # Modular ML pipeline  
├── data/               # Raw and processed data (not committed)   
├── output/             # Generated predictions (not committed)   
├── run_pipeline.py     # End-to-end execution   
├── requirements.txt  
└── README.md

## 🔬 Modeling Approach
1️⃣ Logistic Regression

Interpretable baseline model  
Odds ratios for risk driver interpretation  
AUC: 0.791

2️⃣ Random Forest

Captures nonlinear patterns  
Class imbalance handled via class weighting  
AUC: 0.827

⚖ Threshold Tuning & Business Alignment

Because the dataset is imbalanced, the default 0.5 classification threshold resulted in low recall for defaulters.  
A systematic threshold sweep was performed.  
Selected cutoff: 0.25

Impact:

Recall for defaulters increased from 12.7% → over 30%.   
Tripled high-risk detection relative to default threshold  
Balanced improved risk capture with manageable false positives

This demonstrates how model thresholds directly influence financial tradeoffs between missed defaults and lost revenue opportunities.

📈 Key Results
Model	AUC
Logistic Regression	0.791
Random Forest	0.827

Threshold tuning significantly improved practical default detection performance.

🚀 How to Run

Install dependencies:
...
pip install -r requirements.txt
...

Place dataset in:

data/raw/cs-training.csv


Run:

python run_pipeline.py


Outputs:

Model evaluation metrics  
Risk scores CSV files   
Threshold sweep analysis  

💡 Why This Project Matters

This repository demonstrates:  

Modular, production-style Python code  
Handling of class imbalance  
Threshold tuning aligned with business objectives  
Model comparison (interpretable vs ensemble)  
Translation of ML outputs into lending risk decisions

🔮 Potential Extensions

Cost-sensitive optimization   
Probability calibration  
Cross-validation and hyperparameter tuning  
Deployment as a REST API

