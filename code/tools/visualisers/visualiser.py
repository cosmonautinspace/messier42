import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

rb_train_losses = np.loadtxt("documentation/visualizations/source/ann_PyBrain/rb_train_losses.csv", delimiter=",")
rb_test_losses = np.loadtxt("documentation/visualizations/source/ann_PyBrain/rb_test_losses.csv", delimiter=",")
g_train_losses = np.loadtxt("documentation/visualizations/source/ann_PyBrain/g_train_losses.csv", delimiter=",")
g_test_losses = np.loadtxt("documentation/visualizations/source/ann_PyBrain/g_test_losses.csv", delimiter=",")

rb_residuals = np.loadtxt("documentation/visualizations/source/ols/rb_ols_residuals.csv", delimiter=",")
g_residuals = np.loadtxt("documentation/visualizations/source/ols/g_ols_residuals.csv", delimiter=",")

plt.figure(figsize=(10, 6))
plt.plot(rb_train_losses, label="RB Train Loss")
plt.plot(rb_test_losses, label="RB Test Loss")
plt.plot(g_train_losses, label="G Train Loss")
plt.plot(g_test_losses, label="G Test Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Training and Testing Loss Curves")
plt.savefig("documentation/visualizations/training_testing_curves.png")

plt.figure(figsize=(10, 6))
plt.hist(rb_residuals, bins=30, alpha=0.5, label="RB Residuals")
plt.hist(g_residuals, bins=30, alpha=0.5, label="G Residuals")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.legend()
plt.title("Residual Diagnostic Plots")
plt.savefig("documentation/visualizations/residual_diagnostic_plots.png")

plt.figure(figsize=(10, 6))
plt.scatter(range(len(rb_residuals)), rb_residuals, alpha=0.5, label="RB Residuals")
plt.scatter(range(len(g_residuals)), g_residuals, alpha=0.5, label="G Residuals")
plt.xlabel("Index")
plt.ylabel("Residuals")
plt.legend()
plt.title("Residual Scatter Plots")
plt.savefig("documentation/visualizations/residual_scatter_plots.png")
