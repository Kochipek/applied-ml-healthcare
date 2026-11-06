"""
Applied Machine Learning in Healthcare
Author: Ipek Kochisarli

This script runs the complete evaluation across three clinical datasets:
1. Cardiotocography (CTG) - Fetal state multi-class classification (SVM vs Random Forest)
2. Pima Indians Diabetes - Unsupervised metabolic phenotyping (K-Means & PCA)
3. Medical Cost Personal Datasets - Healthcare expense prediction (Linear vs Random Forest)
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (confusion_matrix, accuracy_score, silhouette_score,
                             davies_bouldin_score, mean_absolute_error,
                             mean_squared_error, r2_score)

RANDOM_STATE = 42
os.makedirs("figures", exist_ok=True)


# ==============================================================================
# 1. Classification: Fetal State Prediction (Cardiotocography)
# ==============================================================================
def evaluate_classification():
    print("\n" + "=" * 60)
    print("1. CARDIOTOCOGRAPHY (CTG) - FETAL STATE CLASSIFICATION")
    print("=" * 60)

    df = pd.read_csv("data/cardiotocography.csv")
    print(f"Total samples: {df.shape[0]}, Features: {df.shape[1] - 1}, Missing values: {df.isnull().sum().sum()}")

    class_names = {1: "Normal", 2: "Suspect", 3: "Pathologic"}
    print("\nClass distribution (NSP):")
    print(df["NSP"].value_counts().sort_index().rename(class_names))

    X = df.drop("NSP", axis=1)
    y = df["NSP"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    pipe_svm = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=10.0, gamma="scale", class_weight="balanced", random_state=RANDOM_STATE)
    )

    pipe_rf = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced", random_state=RANDOM_STATE)
    )

    pipe_svm.fit(X_train, y_train)
    pipe_rf.fit(X_train, y_train)

    y_pred_svm = pipe_svm.predict(X_test)
    y_pred_rf = pipe_rf.predict(X_test)

    print(f"\nSVM Test Accuracy           : {accuracy_score(y_test, y_pred_svm)*100:.2f}%")
    print(f"Random Forest Test Accuracy : {accuracy_score(y_test, y_pred_rf)*100:.2f}%")

    labels = ["Normal (1)", "Suspect (2)", "Pathologic (3)"]

    def compute_class_metrics(cm, model_name):
        total = np.sum(cm)
        rows = []
        for i, label in enumerate(labels):
            tp = cm[i, i]
            fn = np.sum(cm[i, :]) - tp
            fp = np.sum(cm[:, i]) - tp
            tn = total - (tp + fn + fp)
            sensitivity = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0
            specificity = (tn / (tn + fp)) * 100 if (tn + fp) > 0 else 0
            rows.append({
                "Class": label, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                "Sensitivity (%)": round(sensitivity, 2),
                "Specificity (%)": round(specificity, 2)
            })
        print(f"\n[{model_name} Per-Class Performance]")
        print(pd.DataFrame(rows).to_string(index=False))

    cm_svm = confusion_matrix(y_test, y_pred_svm)
    cm_rf = confusion_matrix(y_test, y_pred_rf)

    compute_class_metrics(cm_svm, "SVM")
    compute_class_metrics(cm_rf, "Random Forest")

    # 10-Fold Stratified Cross Validation
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    cv_svm = cross_val_score(pipe_svm, X, y, cv=skf, scoring="accuracy")
    cv_rf = cross_val_score(pipe_rf, X, y, cv=skf, scoring="accuracy")

    print(f"\n10-Fold CV SVM Mean Accuracy           : {cv_svm.mean()*100:.2f}% (+/- {cv_svm.std()*100:.2f}%)")
    print(f"10-Fold CV Random Forest Mean Accuracy : {cv_rf.mean()*100:.2f}% (+/- {cv_rf.std()*100:.2f}%)")

    # Plot Confusion Matrices
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, cm, title, cmap in zip(axes, [cm_svm, cm_rf], ["SVM (RBF) Confusion Matrix", "Random Forest Confusion Matrix"], ["Blues", "Greens"]):
        sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, cbar=False,
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("True Class")
    plt.tight_layout()
    plt.savefig("figures/ctg_confusion_matrices.png", dpi=150)
    plt.close()

    # Feature Importance Plot
    rf = pipe_rf.named_steps["randomforestclassifier"]
    importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
    plt.figure(figsize=(8, 5.5))
    importance.plot(kind="barh", color="#2b5c8f")
    plt.title("Random Forest - Feature Importance (CTG)", fontsize=12, fontweight="bold")
    plt.xlabel("Mean Decrease in Impurity (Gini)")
    plt.tight_layout()
    plt.savefig("figures/ctg_feature_importance.png", dpi=150)
    plt.close()


# ==============================================================================
# 2. Clustering: Metabolic Phenotyping (Diabetes)
# ==============================================================================
def evaluate_clustering():
    print("\n" + "=" * 60)
    print("2. PIMA INDIANS DIABETES - K-MEANS CLUSTERING")
    print("=" * 60)

    df = pd.read_csv("data/diabetes.csv")
    cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
    X = df[cols].copy()

    zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in zero_cols:
        X[col] = X[col].replace(0, X.loc[X[col] > 0, col].median())

    X_scaled = StandardScaler().fit_transform(X)

    print("Clustering Metrics across k (2 to 6):")
    k_range = range(2, 7)
    metrics = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        db = davies_bouldin_score(X_scaled, labels)
        metrics.append({"k": k, "Silhouette": sil, "Davies-Bouldin": db})
        print(f"k={k} -> Silhouette Score: {sil:.4f} | Davies-Bouldin Index: {db:.4f}")

    metrics_df = pd.DataFrame(metrics).set_index("k")

    # Plot Metrics
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(metrics_df.index, metrics_df["Silhouette"], "o-", color="#1f77b4", label="Silhouette Score")
    ax1.set_xlabel("Number of Clusters (k)")
    ax1.set_ylabel("Silhouette Score", color="#1f77b4")
    ax2 = ax1.twinx()
    ax2.plot(metrics_df.index, metrics_df["Davies-Bouldin"], "s--", color="#d62728", label="Davies-Bouldin")
    ax2.set_ylabel("Davies-Bouldin Index", color="#d62728")
    plt.title("K-Means Cluster Validity Metrics", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("figures/diabetes_clustering_metrics.png", dpi=150)
    plt.close()

    # k=2 Phenotyping & PCA
    km2 = KMeans(n_clusters=2, n_init=10, random_state=RANDOM_STATE)
    df["Cluster"] = km2.fit_predict(X_scaled)

    profile = df.groupby("Cluster")[["Glucose", "BMI", "Age", "Insulin"]].mean().round(1)
    profile["Sample Count"] = df["Cluster"].value_counts().sort_index()
    profile["True Diabetes Rate (%)"] = (df.groupby("Cluster")["Outcome"].mean() * 100).round(1)
    print("\nCluster Phenotypes (k=2):")
    print(profile)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    plt.figure(figsize=(7, 5))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df["Cluster"], cmap="coolwarm", s=22, alpha=0.7)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
    plt.title("K-Means (k=2) Patient Clusters on First 2 PCs", fontsize=12, fontweight="bold")
    plt.colorbar(scatter, ticks=[0, 1], label="Cluster")
    plt.tight_layout()
    plt.savefig("figures/diabetes_pca_clusters.png", dpi=150)
    plt.close()


# ==============================================================================
# 3. Regression: Healthcare Expense Prediction
# ==============================================================================
def evaluate_regression():
    print("\n" + "=" * 60)
    print("3. MEDICAL COST PERSONAL DATASETS - REGRESSION ANALYSIS")
    print("=" * 60)

    df = pd.read_csv("data/insurance.csv")
    df_enc = pd.get_dummies(df, drop_first=True)
    X = df_enc.drop("charges", axis=1)
    y = df_enc["charges"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )

    lr = LinearRegression().fit(X_train, y_train)
    rf = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE).fit(X_train, y_train)

    pred_lr = lr.predict(X_test)
    pred_rf = rf.predict(X_test)

    results = []
    for name, pred in [("Multiple Linear Regression", pred_lr), ("Random Forest Regressor", pred_rf)]:
        mae = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        r2 = r2_score(y_test, pred)
        results.append({"Model": name, "MAE ($)": round(mae, 2), "RMSE ($)": round(rmse, 2), "R²": round(r2, 4)})

    print("\nModel Benchmark Comparison:")
    print(pd.DataFrame(results).to_string(index=False))

    # Plot Actual vs Predicted
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, (name, pred) in zip(axes, [("Linear Regression", pred_lr), ("Random Forest Regressor", pred_rf)]):
        ax.scatter(y_test, pred, s=18, alpha=0.6, color="#1f77b4" if "Linear" in name else "#2ca02c")
        lims = [y_test.min(), y_test.max()]
        ax.plot(lims, lims, "r--", lw=1.5, label="y = x (Perfect Fit)")
        ax.set_title(f"{name} (R² = {r2_score(y_test, pred):.3f})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Actual Charges ($)")
        ax.legend(loc="upper left")
    axes[0].set_ylabel("Predicted Charges ($)")
    plt.tight_layout()
    plt.savefig("figures/medical_cost_regression.png", dpi=150)
    plt.close()

    print("\nInteraction Effect: Average Charges by Smoking and Obesity (BMI >= 30):")
    interaction = df.assign(obese=df["bmi"] >= 30).groupby(["smoker", "obese"])["charges"].mean().round(2)
    print(interaction)
    print()


if __name__ == "__main__":
    evaluate_classification()
    evaluate_clustering()
    evaluate_regression()
