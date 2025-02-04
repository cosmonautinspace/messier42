import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

train_losses = np.loadtxt("documentation/visualisations/source/ann_PyBrain/train_losses.csv", delimiter=",")
test_losses = np.loadtxt("documentation/visualisations/source/ann_PyBrain/test_losses.csv", delimiter=",")

rb_residuals = np.loadtxt("documentation/visualisations/source/ols/rb_ols_residuals.csv", delimiter=",")
g_residuals = np.loadtxt("documentation/visualisations/source/ols/g_ols_residuals.csv", delimiter=",")

plt.figure(figsize=(10, 6))
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Training and Testing Loss Curves")
plt.savefig("documentation/visualisations/training_testing_curves.png")

plt.figure(figsize=(10, 6))
plt.hist(rb_residuals, bins=30, alpha=0.5, label="RB Residuals")
plt.hist(g_residuals, bins=30, alpha=0.5, label="G Residuals")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.legend()
plt.title("Residual Diagnostic Plots")
plt.savefig("documentation/visualisations/residual_diagnostic_plots.png")

plt.figure(figsize=(10, 6))
plt.scatter(range(len(rb_residuals)), rb_residuals, alpha=0.5, label="RB Residuals")
plt.scatter(range(len(g_residuals)), g_residuals, alpha=0.5, label="G Residuals")
plt.xlabel("Index")
plt.ylabel("Residuals")
plt.legend()
plt.title("Residual Scatter Plots")
plt.savefig("documentation/visualisations/residual_scatter_plots.png")
