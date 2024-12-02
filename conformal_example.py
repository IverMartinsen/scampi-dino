import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

std = 0.1
mean_c0 = 0.25
mean_c1 = 1.50
mean_ood = 1.25
n = 100000
alpha = 0.100
means = np.array([mean_c0, mean_c1, mean_ood])
freq_dist = np.array([0.15, 0.15, 0.70])
acc = 1.00

# get the theortical normal quantiles
q0 = norm.ppf(1 - alpha, loc=mean_c0, scale=std)
q1 = norm.ppf(1 - alpha, loc=mean_c1, scale=std)
q = np.array([q0, q1])




y_true = np.random.choice([0, 1, 2], n, p=freq_dist)
x_new = np.random.normal(means[y_true], std)
correct_pred = np.random.choice([0, 1], n, p=[1 - acc, acc])
y_pred = np.zeros_like(y_true)
for c in [0, 1, 2]:
    idx = np.where(y_true == c)[0]
    if c == 0:
        y_pred[idx] = 1 - correct_pred[idx]
    elif c == 1:
        y_pred[idx] = correct_pred[idx]
    else:
        y_pred[idx] = np.random.choice([0, 1], len(idx), p=[0.5, 0.5])

y_pred[np.where(x_new > q[y_pred])[0]] = 2 # label as OOD

# true and false positive rates
tpr = (y_true == 2)[y_pred == 2].mean()
fpr = (y_true != 2)[y_pred == 2].mean()
fnr = (y_true == 2)[y_pred != 2].mean()
print(f'tpr: {tpr}, fpr: {fpr}', f'fnr: {fnr}')
# false positive rates for class 0 and 1
fpr_c0 = (y_pred == 2)[np.where(y_true == 0)[0]].mean()
fpr_c1 = (y_pred == 2)[np.where(y_true == 1)[0]].mean()
print(f'fpr_c0: {fpr_c0}, fpr_c1: {fpr_c1}')
