import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

runs_to_plot = [
    "dino-v1-7374281",
    "dino-v1-7374283",
    "dino-v1-7374301",
    "dino-v1-7261890",
    "dino-v1-7265199",
    "dino-v1-7265452",
    "dino-v1-7267359",
]    

path = "./trained_models/LUMI/zip scrapings (huge)/"
paths = [os.path.join(path, run, "summary_metrics.json") for run in runs_to_plot]

data = []
for path in paths:
    with open(path, "r") as f:
        data.append(json.load(f))

labels = {
    'log_loss': "Log loss",
    'cbir_accuracy': "precision@k",
    'nn_f1': "F1 score",
}

plt.figure(figsize=(10, 5))

for metric in ['log_loss', 'cbir_accuracy', 'nn_f1']:
    x = [d['dataset_size'] for d in data]
    y = [d[metric] for d in data]
    df = pd.DataFrame({'x': x, 'y': y})
    # compute mean across each x value
    x = df.groupby('x').mean().index
    y = df.groupby('x').mean()['y']
    label = labels[metric]
    plt.plot(x, y, label=label, marker='o')

plt.xticks([d['dataset_size'] for d in data], rotation=45, fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel("Dataset size", fontsize=16)
plt.ylabel("Metric value", fontsize=16)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig("dataset_size_default_params.png", dpi=300)
plt.close()




runs_to_plot = [
    "dino-v1-7374281",
    "dino-v1-7374283",
    "dino-v1-7374301",
    "dino-v1-7261890",
    "dino-v1-7395482",
    "dino-v1-8370995",
]    

path = "./trained_models/LUMI/zip scrapings (huge)/"
paths = [os.path.join(path, run, "summary_metrics.json") for run in runs_to_plot]

data = []
for path in paths:
    with open(path, "r") as f:
        data.append(json.load(f))

plt.figure(figsize=(10, 5))

for metric in ['log_loss', 'cbir_accuracy', 'nn_f1']:
    x = [d['dataset_size'] for d in data]
    y = [d[metric] for d in data]
    df = pd.DataFrame({'x': x, 'y': y})
    # compute mean across each x value
    x = df.groupby('x').mean().index
    y = df.groupby('x').mean()['y']
    label = labels[metric]
    plt.plot(x, y, label=label, marker='o')

plt.xticks([d['dataset_size'] for d in data], rotation=45, fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel("Dataset size", fontsize=16)
plt.ylabel("Metric value", fontsize=16)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig("dataset_size_new_params.png", dpi=300)
plt.close()



















