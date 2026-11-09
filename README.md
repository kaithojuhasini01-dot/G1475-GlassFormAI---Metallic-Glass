# GlassFormAI - Machine Learning Aided Prediction of Glass-Forming Ability of Metallic Glass

## 🚀 Project Overview

This project reproduces and extends the machine learning pipeline from the paper "Machine Learning Aided Prediction of Glass-Forming Ability of Metallic Glass" (Liu et al., 2023). We successfully built predictive models to determine whether an alloy can form a metallic glass and to predict its key thermal properties (Tl, Tx, Tg). 

Beyond just prediction, we performed comprehensive **Model Interpretation** using Feature Importance, Partial Dependence Plots (PDP), and Individual Conditional Expectation (ICE) plots to explain *how* the models make their decisions. Our final ETR regression models achieved an exceptional **R² score > 0.96**, outperforming the base paper's reported results. 

Finally, we used our predicted temperatures to calculate the **Gamma (\(\gamma\)) GFA parameter**, and implemented the advanced **NSGA-II Genetic Algorithm** (executed via Google Colab) to optimize the Zr-Cu-Ni-Al composition, and deployed a **fully functional Interactive Web Tool (GUI)** using Streamlit. This tool allows users to input any chemical formula and instantly receive real-time predictions for thermal properties, GFA parameters, and view optimized alloy compositions.

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
- `documentation/` : Comprehensive logs for data processing, model training, interpretation, GFA calculation, NSGA-II optimization, and the interactive tool.
- `models/` : Saved trained models(.pkl files) for classification and regression.
- `notebooks/` : Jupyter Notebook containing the executed NSGA-II optimization code and outputs.
- `results/` : Model evaluatioon metrics (CSV), predictions, and interpretation plots(PDP/ICE), GFA parameters, and the final NSGA-II optimized compositions.
- `src/` : Python scripts for data processing, feature generation, selection, training, interpretation, GFA calculation, optimization and the Streamlit interactive GUI(`app.py`).

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
- [x] Day 13: Completed Model Interpretation(Feature Importance, PDP, and ICE plots) for all targets.
- [x] Day 14: Calculated GFA parameters (Gamma) using predicted temperatures and saved results.
- [x] Day 15: Implemented and executed NSGA-II optimization (via Colab) to generate novel alloy compositions.
- [x] Day 16: Built and deployed the interactive Streamlit GUI for real-time alloy predictions.

## 🛠️ Technologies Used

**Programming Language & Environment**
- Python 3
- Google Colab / Jupyter Notebook (for executing the NSGA-II optimization)
- VS Code (for core development)
- Streamlit (for building the interactive web GUI)

**Data Manipulation & Processing**
- Pandas, NumPy
- Pymatgen (for parsing chemical formulas into elemental fractions)

**Feature Engineering & Selection**
- Matminer (Magpie preset for generating 140 numerical descriptors)
- Scikit-Learn (Variance filter, F-test, Recursive Feature Elimination)

**Machine Learning & Optimization**
- Scikit-Learn (LabelEncoder, Linear Regression, Data Splitting)
- XGBoost (XGC for Classification)
- Scikit-Learn ExtraTreesRegressor (ETR for Regression)
- Pymoo (NSGA-II Genetic Algorithm for Composition Optimization)

**Model Interpretation & Visualization**
- Matplotlib, Seaborn (Feature Importance, PDP, and ICE plots)

**Model Saving & Deployment**
- Joblib (saving `.pkl` models)

**Version Control**
- Git & GitHub
