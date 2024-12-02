import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

std = 0.1
mean_c0 = 0.50
mean_c1 = 1.00
mean_ood = 1.25
n = 100000
alpha = 0.40
freq_dist = np.array([0.01, 0.01, 0.98])
acc = 0.95

means = np.array([mean_c0, mean_c1, mean_ood])

# get the theortical normal quantiles
q0 = norm.ppf(1 - alpha, loc=mean_c0, scale=std)
q1 = norm.ppf(1 - alpha, loc=mean_c1, scale=std)
q = np.array([q0, q1])



for i in range(10):
    p_error = 1 - norm.cdf((q1, q0), loc=(mean_c0, mean_c1), scale=std)

    alpha_adjusted = (alpha - p_error * (1 - acc)) / acc

    q = norm.ppf(1 - alpha_adjusted, loc=(mean_c0, mean_c1), scale=std)

q0, q1 = q

# plot distributions
plt.figure(figsize=(10, 5))
x = np.linspace(-0.5, 2, 1000)
plt.plot(x, norm.pdf(x, loc=mean_c0, scale=std), label='Class 0')
plt.plot(x, norm.pdf(x, loc=mean_c1, scale=std), label='Class 1')
plt.plot(x, norm.pdf(x, loc=mean_ood, scale=std), label='OOD data')
plt.fill_between(x, norm.pdf(x, loc=mean_c0, scale=std), where=x < q0, alpha=0.5, label='Data not rejected')
plt.fill_between(x, norm.pdf(x, loc=mean_c1, scale=std), where=x < q1, alpha=0.5)
plt.fill_between(x, norm.pdf(x, loc=mean_ood, scale=std), where=x < q1, alpha=0.5)
plt.axvline(q0, color='r', linestyle='--', label='Rejection threshold')
plt.axvline(q1, color='r', linestyle='--')
plt.axis('off')
plt.legend()
plt.title('Distribution of non-conformity scores')
plt.savefig('non_conformity_scores.png', dpi=300)
plt.close()

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

