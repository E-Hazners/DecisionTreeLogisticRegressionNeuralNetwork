# PART 4 - Multilayer Neural Network with Regularization

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.neural_network import MLPClassifier
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
class_names   = np.load("class_names.npy",   allow_pickle=True).tolist()

print("Loaded data:")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Classes: {class_names}")
print()

# 4.1 Train / Test split + feature scaling
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

print("4.1 Train / Test split + StandardScaler")
print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print()

# 4.2 Build baseline MLP
baseline_mlp = MLPClassifier(
# 2 hidden layers
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
# L2 regularisation
    alpha=1e-4,
    max_iter=500,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=15
)

# 4.3 Train and evaluate
baseline_mlp.fit(X_train_s, y_train)


print("4.3 Baseline MLP trained.")
print(f"Hidden layers: {baseline_mlp.hidden_layer_sizes}")
print(f"Actual iterations: {baseline_mlp.n_iter_}")
print()

# 4.4 Model parameters

print("4.4 Model parameters")
for i, (W, b) in enumerate(zip(baseline_mlp.coefs_, baseline_mlp.intercepts_)):
    print(f"Layer {i+1}:")
    print(f"Weight matrix shape: {W.shape}")
    print(f"Bias vector shape: {b.shape}")
    print(f"Weight norm (L2): {np.linalg.norm(W):.4f}")
    print(f"Bias values: {b.round(4)}")
print()

# 4.5 Performance
y_pred_train = baseline_mlp.predict(X_train_s)
y_pred_test  = baseline_mlp.predict(X_test_s)

acc_train = accuracy_score(y_train, y_pred_train)
acc_test  = accuracy_score(y_test,  y_pred_test)

cv_scores = cross_val_score(
    MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu",
                  solver="adam", alpha=1e-4, max_iter=500,
                  random_state=42),
    scaler.transform(X), y, cv=5, scoring="accuracy"
)

print("4.5 Performance")
print(f"Training accuracy: {acc_train:.4f}  ({acc_train*100:.2f} %)")
print(f"Test accuracy: {acc_test:.4f}  ({acc_test*100:.2f} %)")
print(f"Cross-val (5-fold): {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
print()
print("Classification report (test set):")
print(classification_report(y_test, y_pred_test, target_names=class_names))

# 4.6 Effect of max_iter (epochs) on performance

print("4.6 Epoch sweep")

epoch_values = [10, 20, 50, 100, 200, 300, 500, 800, 1000]
epoch_train_acc, epoch_test_acc = [], []

for epochs in epoch_values:
    m = MLPClassifier(
        hidden_layer_sizes=(64, 32), activation="relu",
        solver="adam", alpha=1e-4, max_iter=epochs,
        random_state=42
    )
    m.fit(X_train_s, y_train)
    epoch_train_acc.append(accuracy_score(y_train, m.predict(X_train_s)))
    epoch_test_acc.append( accuracy_score(y_test,  m.predict(X_test_s)))
    print(f"max_iter={epochs:>5d}  train={epoch_train_acc[-1]:.4f}  test={epoch_test_acc[-1]:.4f}")

print()

# 4.7 Effect of depth (layers) and width (neurons per layer)

print("4.7 Architecture sweep (depth x width)")

# number of hidden layers
depth_configs  = [1, 2, 3, 4]
# neurons per layer
width_configs  = [8, 16, 32, 64, 128]

depth_matrix_train = np.zeros((len(depth_configs), len(width_configs)))
depth_matrix_test  = np.zeros((len(depth_configs), len(width_configs)))

for di, depth in enumerate(depth_configs):
    for wi, width in enumerate(width_configs):
        layers = tuple([width] * depth)
        m = MLPClassifier(
            hidden_layer_sizes=layers, activation="relu",
            solver="adam", alpha=1e-4, max_iter=500,
            random_state=42
        )
        m.fit(X_train_s, y_train)
        depth_matrix_train[di, wi] = accuracy_score(y_train, m.predict(X_train_s))
        depth_matrix_test[di, wi]  = accuracy_score(y_test,  m.predict(X_test_s))
        print(f"layers={depth}  neurons={width:>3d}  "
              f"train={depth_matrix_train[di,wi]:.4f}  "
              f"test={depth_matrix_test[di,wi]:.4f}")

print()


# 4.8 Effect of regularisation (L1-like / L2 / Dropout simulation)
print("4.8 Regularisation sweep")

alpha_values = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0]
reg_train_acc, reg_test_acc = [], []

for alpha in alpha_values:
    m = MLPClassifier(
        hidden_layer_sizes=(64, 32), activation="relu",
        solver="adam", alpha=alpha, max_iter=500,
        random_state=42
    )
    m.fit(X_train_s, y_train)
    reg_train_acc.append(accuracy_score(y_train, m.predict(X_train_s)))
    reg_test_acc.append( accuracy_score(y_test,  m.predict(X_test_s)))
    print(f"alpha={alpha:<8}  train={reg_train_acc[-1]:.4f}  test={reg_test_acc[-1]:.4f}")

print()

# Figures

# Figure 1: Training curve of baseline MLP
fig1, ax = plt.subplots(figsize=(8, 4))
ax.plot(baseline_mlp.loss_curve_, label="Training loss", color="steelblue")
if hasattr(baseline_mlp, "validation_scores_") and baseline_mlp.validation_scores_:
    ax.plot(baseline_mlp.validation_scores_,  label="Validation accuracy",
            color="darkorange", linestyle="--")
ax.set_xlabel("Iteration (epoch)")
ax.set_ylabel("Loss / Accuracy")
ax.set_title("Baseline MLP - Loss Curve During Training")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("part4_fig1_loss_curve.png", dpi=150)
plt.show()

# Figure 2: Confusion matrix (baseline)
fig2, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred_test)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names).plot(
    cmap="Blues", ax=ax
)
ax.set_title(f"Baseline MLP - Confusion Matrix\n(test accuracy = {acc_test*100:.2f} %)")
plt.tight_layout()
plt.savefig("part4_fig2_confusion.png", dpi=150)
plt.show()

# Figure 3 (4.6): Epoch sweep
fig3, ax = plt.subplots(figsize=(9, 4))
ax.plot(epoch_values, epoch_train_acc, "o-", label="Train accuracy", color="steelblue")
ax.plot(epoch_values, epoch_test_acc,  "s--", label="Test accuracy",  color="darkorange")
ax.set_xlabel("max_iter (epochs)")
ax.set_ylabel("Accuracy")
ax.set_title("4.6  Effect of Epochs (max_iter) on MLP Accuracy")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("part4_fig3_epoch_sweep.png", dpi=150)
plt.show()

# Figure 4 (4.7): Depth x Width heatmaps
fig4, axes = plt.subplots(1, 2, figsize=(14, 5))
for mat, title, ax in zip(
    [depth_matrix_train, depth_matrix_test],
    ["Train Accuracy", "Test Accuracy"],
    axes
):
    im = ax.imshow(mat, vmin=0.4, vmax=1.0, cmap="YlGn", aspect="auto")
    ax.set_xticks(range(len(width_configs)))
    ax.set_xticklabels(width_configs)
    ax.set_yticks(range(len(depth_configs)))
    ax.set_yticklabels([f"{d} layers" for d in depth_configs])
    ax.set_xlabel("Neurons per layer (width)")
    ax.set_ylabel("Number of hidden layers (depth)")
    ax.set_title(f"4.7  Depth x Width - {title}")
    plt.colorbar(im, ax=ax)
    for i in range(len(depth_configs)):
        for j in range(len(width_configs)):
            ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                    fontsize=9, color="black")
plt.tight_layout()
plt.savefig("part4_fig4_architecture_heatmap.png", dpi=150)
plt.show()

print("Figures saved")