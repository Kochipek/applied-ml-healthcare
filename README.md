# Applied Machine Learning in Healthcare
### Clinical Benchmarks: Fetal State Classification, Metabolic Phenotyping, and Cost Prediction

This repository implements applied machine learning workflows across three distinct clinical datasets, addressing supervised classification under class imbalance, unsupervised patient phenotyping, and non-linear medical expense regression.

---

## Benchmark Summary

| Domain | Task | Model | Primary Metrics | Key Finding |
|---|---|---|---|---|
| **Fetal Cardiotocography** | Multi-class Classification (`Normal`, `Suspect`, `Pathologic`) | Random Forest & SVM (RBF) | **93.93% $\pm$ 1.30%** 10-Fold CV Acc<br>**85.71%** Pathologic Sensitivity | Successfully detects 30/35 critical fetal distress cases with 99.2% specificity. Heart rate variability (`ASTV`, `ALTV`) is the dominant biomarker. |
| **Pima Indians Diabetes** | Unsupervised Patient Clustering | K-Means ($k=2..6$) & PCA | **0.2045** Silhouette Score ($k=2$)<br>Outcome separation: **56.4% vs 18.2%** | Without access to diagnosis labels, clusters split patients into a high-risk obese phenotype vs. a low-risk normoglycemic cohort. |
| **Medical Expenses** | Continuous Cost Regression | Random Forest Regressor & Multiple Linear Regression | **$2,550** MAE / **0.865** $R^2$<br>(vs $4,181 MAE / 0.784 $R^2$) | Random Forest reduces prediction error by 39% by capturing the non-linear compounding interaction between smoking and obesity. |

---

## Analysis & Methodology

### 1. Fetal State Classification (Cardiotocography)
- **Dataset:** 2,126 fetal heart rate recordings with 21 morphological and statistical features from the UCI Machine Learning Repository.
- **Handling Class Imbalance:** With only 8.3% Pathologic samples, models are trained using `class_weight='balanced'` inside `sklearn.pipeline.Pipeline` with `StandardScaler` to prevent data leakage.
- **Evaluation:** Evaluated using confusion matrices, per-class sensitivity (recall), specificity, and 10-fold stratified cross-validation. Both SVM and Random Forest achieved **85.71% sensitivity** on Pathologic cases, minimizing dangerous false negatives.

### 2. Unsupervised Metabolic Phenotyping (Diabetes)
- **Dataset:** 768 patient records from the Pima Indians Diabetes dataset.
- **Clinical Zero Imputation:** Biologically impossible zero values in physiological markers (`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`) were imputed using non-zero median values prior to standardization.
- **Cluster Validation:** Evaluated for $k \in [2, 6]$ using Silhouette Score and Davies-Bouldin Index. $k=2$ provided the strongest separation, cleanly segregating patients into high-risk (56.4% true diabetes rate) and low-risk (18.2% true diabetes rate) phenotypes.

### 3. Healthcare Expenditure Regression (Medical Costs)
- **Dataset:** 1,338 US health insurance beneficiaries with 6 demographic and behavioral predictors.
- **Non-Linear Compounding Effects:** In non-smokers, obesity increases average annual medical charges moderately (from $7,977 to $8,843). However, in smokers, obesity multiplies annual costs from $21,363 to **$41,558**. Linear models fail to capture this threshold effect, whereas Random Forest achieves an $R^2$ of **0.865**.

---

## Project Structure

```text
.
├── healthcare_ml_analysis.ipynb   # Interactive Jupyter notebook with complete outputs and explanations
├── main.py                        # Standalone evaluation script to run all models and generate plots
├── data/
│   ├── cardiotocography.csv       # UCI CTG dataset (2,126 samples)
│   ├── diabetes.csv               # Pima Indians Diabetes dataset (768 samples)
│   └── insurance.csv              # Medical Cost Personal Datasets (1,338 samples)
├── figures/                       # High-resolution benchmark figures
│   ├── ctg_confusion_matrices.png
│   ├── ctg_feature_importance.png
│   ├── diabetes_clustering_metrics.png
│   ├── diabetes_pca_clusters.png
│   └── medical_cost_regression.png
├── requirements.txt               # Required Python packages
└── README.md
```

---

## Getting Started

### 1. Clone & Install Dependencies

```bash
git clone https://github.com/Kochipek/applied-ml-healthcare.git
cd applied-ml-healthcare
pip install -r requirements.txt
```

### 2. Run the Analysis Script

To execute all models, print evaluation metrics to the console, and update the figures:

```bash
python main.py
```

### 3. Interactive Jupyter Notebook

To explore the step-by-step analysis and interactive visualizations:

```bash
jupyter notebook healthcare_ml_analysis.ipynb
```

---

## References

1. **Cardiotocography Data Set**, UCI Machine Learning Repository, Faculty of Medicine and INEB, University of Porto.
2. **Pima Indians Diabetes Database**, National Institute of Diabetes and Digestive and Kidney Diseases, Kaggle.
3. **Medical Cost Personal Datasets**, M. Choi, Kaggle.
4. **Scikit-learn: Machine Learning in Python**, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
