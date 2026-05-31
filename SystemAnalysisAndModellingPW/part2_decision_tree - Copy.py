# PART 2 - Decision Tree Creation and Performance Evaluation


import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay
)

# Load prepared data (produced by part1_data_preparation.py)
X = np.load("X_data.npy")
y = np.load("y_data.npy")
feature_names = np.load("feature_names.npy", allow_pickle=True).tolist()
class_names = np.load("class_names.npy",   allow_pickle=True).tolist()

print("Loaded data:")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Classes: {class_names}")
print()


# 2.1 Train / Test split  (75 % / 25 %)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("2.1 Train / Test split")
print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print()


# 2.2 & 2.3 Build and train Decision Tree (ID3-like: information gain / CART-like: Gini impurity).
clf = DecisionTreeClassifier(
    # 2.4 metric: gini impurity (default)
    criterion="gini",      
    random_state=42
)

clf.fit(X_train, y_train)

print("2.2 / 2.3  Decision Tree trained.")
print()


# 2.4 Tree structure details
print("2.4 Tree structure")
print(f"Number of leaves : {clf.get_n_leaves()}")
print(f"Tree depth: {clf.get_depth()}")
print(f"Criterion (metric): {clf.criterion}")
print()

# Text representation of the tree (first 4 levels)
print("Text representation (depth <= 4):")
print(export_text(clf, feature_names=feature_names, max_depth=4))


# 2.5 Performance on training and test sets
y_pred_train = clf.predict(X_train)
y_pred_test  = clf.predict(X_test)

acc_train = accuracy_score(y_train, y_pred_train)
acc_test  = accuracy_score(y_test,  y_pred_test)

# 5 fold cross-validation on the full dataset
cv_scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")

print("2.5  Performance")
print(f"Training accuracy: {acc_train:.4f}  ({acc_train*100:.2f} %)")
print(f"Test accuracy: {acc_test:.4f}  ({acc_test*100:.2f} %)")
print(f"Cross-val (5-fold): {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
print()
print("Classification report (test set):")
print(classification_report(y_test, y_pred_test, target_names=class_names))


# 2.6 Graphical visualisation
fig, axes = plt.subplots(1, 2, figsize=(22, 8))

# Panel 1: Full tree diagram -
plot_tree(
    clf,
    feature_names=feature_names,
    class_names=class_names,
    filled=True,
    rounded=True,
    max_depth=4, # limit display depth for readability
    fontsize=8,
    ax=axes[0]
)
axes[0].set_title(
    f"Decision Tree (criterion={clf.criterion}, depth={clf.get_depth()}, "
    f"leaves={clf.get_n_leaves()})\n[display capped at depth 4]",
    fontsize=11
)

# Panel 2: Confusion matrix (test set) -
cm = confusion_matrix(y_test, y_pred_test)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names).plot(
    cmap="Blues", ax=axes[1]
)
axes[1].set_title(
    f"Confusion Matrix - Test Set\n(accuracy = {acc_test*100:.2f} %)",
    fontsize=11
)

plt.tight_layout()
plt.savefig("part2_decision_tree.png", dpi=150, bbox_inches="tight")
plt.show()
print()
print("Figure saved: part2_decision_tree.png")