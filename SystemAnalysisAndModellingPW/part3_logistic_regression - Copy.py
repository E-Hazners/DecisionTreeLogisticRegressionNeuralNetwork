# PART 3 - Logistic Regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay
)

# Load prepared data
X = np.load("X_data.npy")
y = np.load("y_data.npy")
feature_names = np.load("feature_names.npy", allow_pickle=True).tolist()
class_names = np.load("class_names.npy",   allow_pickle=True).tolist()

print("Loaded data:")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Classes: {class_names}")
print()

# 3.1 Train / Test split  (75 % / 25 %) + feature scaling

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

scaler  = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

print("3.1 Train / Test split + StandardScaler")
print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print()


# 3.2 Build Logistic Regression model
lr = LogisticRegression(
    solver="lbfgs",    # lbfgs handles multinomial natively in modern sklearn
    max_iter=1000,
    random_state=42
)

# 3.3 Train on training set, evaluate on test set
lr.fit(X_train_s, y_train)

print("3.3 Model trained")
print()

# 3.4 sModel parameters
print("3.4 Model parameters")
print(f"Intercept (one value per class):  {lr.intercept_}")
print()

coeff_df = pd.DataFrame(
    lr.coef_,
    columns=feature_names,
    index=[f"Class {c}" for c in class_names]
)
print("Coefficients (rows = classes, columns = features):")
print(coeff_df.to_string())
print()

# 3.5 Performance on training and test sets
y_pred_train = lr.predict(X_train_s)
y_pred_test  = lr.predict(X_test_s)

acc_train = accuracy_score(y_train, y_pred_train)
acc_test  = accuracy_score(y_test,  y_pred_test)

# 5-fold cross-validation
cv_scores = cross_val_score(lr, scaler.transform(X), y, cv=5, scoring="accuracy")

print("3.5 Performance")
print(f"Training accuracy: {acc_train:.4f}  ({acc_train*100:.2f} %)")
print(f"Test accuracy: {acc_test:.4f}  ({acc_test*100:.2f} %)")
print(f"Cross-val (5-fold): {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
print()
print("Classification report (test set):")
print(classification_report(y_test, y_pred_test, target_names=class_names))

# Visualisation of two panels
# 1.Coefficient heatmap
# 2.Confusion matrix
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Panel 1:Coefficient heatmap -
im = axes[0].imshow(lr.coef_, aspect="auto", cmap="RdBu_r")
axes[0].set_xticks(range(len(feature_names)))
axes[0].set_xticklabels(feature_names, rotation=45, ha="right", fontsize=8)
axes[0].set_yticks(range(len(class_names)))
axes[0].set_yticklabels([f"Class {c}" for c in class_names], fontsize=9)
axes[0].set_title("Logistic Regression - Coefficient Heatmap", fontsize=11)
plt.colorbar(im, ax=axes[0])

for i in range(lr.coef_.shape[0]):
    for j in range(lr.coef_.shape[1]):
        axes[0].text(j, i, f"{lr.coef_[i, j]:.2f}",
                     ha="center", va="center", fontsize=6,
                     color="white" if abs(lr.coef_[i, j]) > 1.5 else "black")

# Panel 2:Confusion matrix -
cm = confusion_matrix(y_test, y_pred_test)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names).plot(
    cmap="Blues", ax=axes[1]
)
axes[1].set_title(
    f"Confusion Matrix - Test Set\n(accuracy = {acc_test*100:.2f} %)",
    fontsize=11
)

plt.tight_layout()
plt.savefig("part3_logistic_regression.png", dpi=150, bbox_inches="tight")
plt.show()
print()
print("Figure saved: part3_logistic_regression.png")