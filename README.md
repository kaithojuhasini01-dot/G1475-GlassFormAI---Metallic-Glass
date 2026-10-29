# GlassFormAI - Machine Learning Aided Prediction of Glass-Forming Ability of Metallic Glass

## 📌 Project Overview
This project aims to reproduce and extend the machine learning pipeline from the paper "Machine Learning Aided Prediction of Glass-Forming Ability of Metallic Glass" (Liu et al., 2023). We are building predictive models to determine whether an alloy can form a metallic glass and to predict its key thermal properties.

## 👥 Team Members
- Hasini Kaithoju (26P85A0502)
- Venu (26P85A0507)
- Abhigna (26P85A0505)
- Krishna (26P85A0513)
- Spoorthi (26P85A0511)
- Laya (26P85A6603)

**Guide:** Mr. P. Sreenivas

## 📂 Repository Structure
- `data/` : Contains raw, cleaned, features and ml_datasets (train/test splits).
- `documentation/` : Daily logs, reports, feature selection analysis and screenshots/evidence.
- `models/` : Saved trained models(.pkl files) for classification and regression.
- `results/` : Model evaluatioon metrics (CSV) and actual vs. predicted outputs.
- `src/` : Python scripts for data processing, feature generation, feature selection, and model training. 

## 🚀 Progress Tracker
- [x] Milestone 1: Data processing and elemental fraction generation.
- [x] Day 2:  Generated 140 numerical features using Matminer.
- [x] Day 3: Fixed the NaN Error in numerical feature generation.
- [x] Day 5: Documented feature comparison between paper and matminer.
- [x] Day 6: Saved final feature dataset and separated targets.
- [x] Day 7: Prepared target datasets.
- [x] Day 8: Prepared target datasets (Classification & Regression) and fixed data leakage.
- [x] Day 9: Completed Feature Selection(Variance,F-test,Embedded,RFE,Exhaustive Search) for Tl,Tx, and Tg.
- [x] Day 10: Prepared Machine Learning datasets (80/20 Train-Test split) for Regression models.
- [x] Day 11: Implemented and saved XGBoost Classification model.Documented dataset limitations.
- [x] Day 12: Trained and saved ETR Regression models(R^2 > 0.96), saved predictions, and all documentation logs.

## 🛠️ Technologies Used
- **Programming Language:** Python 3
- **Data Manipulation:** Pandas, NumPy
- **Materials Science & Feature Engineering:** Pymatgen, Matminer
- **Machine Learning & Evaluation:** Scikit-Learn, XGBoost(Classification), ETR(Regression)
- **Model Saving & Deployment:** Joblib
- **Version Control:** Git & GitHub
