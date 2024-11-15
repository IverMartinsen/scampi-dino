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

plt.xticks([d['dataset_size'] for d in data], rotation=45)
plt.xlabel("Dataset size")
plt.ylabel("Metric value")
plt.legend()
plt.tight_layout()
plt.savefig("dataset_size_default_params.png", dpi=300)
plt.close()





runs_to_plot = [
    "dino-v1-7261890",
    "dino-v1-8013040",
    "dino-v1-7265199",
    "dino-v1-7265452",
    "dino-v1-8013053",
    "dino-v1-8350705",
    "dino-v1-7267359",
    "dino-v1-8025721",
    "dino-v1-8025747",
]    

path = "./trained_models/LUMI/zip scrapings (huge)/"
paths = [os.path.join(path, run, "summary_metrics.json") for run in runs_to_plot]

data = []
for path in paths:
    with open(path, "r") as f:
        data.append(json.load(f))

for d in data:
    d["alpha"] = np.round((1 - d["momentum"]) * d["dataset_size"] * 256 / 768640, 1)
    d["log_loss"] -= 0.4
    print(d["dataset_size"], d["momentum"], d["alpha"])

linestyles = ['-', '--', ':']
cmap = {768640: 'tab:blue', 1537280: 'tab:orange', 3074560: 'tab:green'}

plt.figure(figsize=(10, 5))

for i, metric in enumerate(['log_loss', 'cbir_accuracy', 'nn_f1']):
    x = [d['alpha'] for d in data]
    y = [d[metric] for d in data]
    z = [d['dataset_size'] for d in data]
    df = pd.DataFrame({'x': x, 'y': y, 'z': z})
    # compute mean across each x value
    z = df.groupby(['x', 'z']).mean().index.get_level_values('z')
    x = df.groupby(['x', 'z']).mean().index.get_level_values('x')
    y = df.groupby(['x', 'z']).mean()['y']
    
    plt.plot([], [], label=labels[metric], color='black', linestyle=linestyles[i])
    
    for z_val in np.unique(z):
        plt.plot(x[z == z_val], y[z == z_val], marker='o', linestyle=linestyles[i], color=cmap[z_val])
        if i == 2:
            plt.plot([], [], label=f"Dataset size: {z_val}", color=cmap[z_val])

plt.xlabel("Alpha")
plt.ylabel("Metric value")
plt.yticks([])
plt.legend()
plt.tight_layout()
plt.savefig("alpha_params.png", dpi=300)
plt.close()



runs_to_plot = [
    "dino-v1-7395482",
    "dino-v1-8350705",
    "dino-v1-8025747",
    "dino-v1-8370995",
]    

path = "./trained_models/LUMI/zip scrapings (huge)/"
paths = [os.path.join(path, run, "summary_metrics.json") for run in runs_to_plot]

data = []
for path in paths:
    with open(path, "r") as f:
        data.append(json.load(f))

plt.figure(figsize=(10, 5))

for i, metric in enumerate(['log_loss', 'cbir_accuracy', 'nn_f1']):
    x = np.array([d['learning_rate'] for d in data])
    y = np.array([d[metric] for d in data])
    z = np.array([d['dataset_size'] for d in data])
    for z_val in np.sort(np.unique(z)):
        plt.plot(x[z == z_val], y[z == z_val], marker='o', label=f"{labels[metric]}, {z_val} images", color=cmap[z_val], linestyle=linestyles[i])

plt.xlabel("Learning rate")
plt.xticks([0.00025, 0.0005])
plt.ylabel("Metric value")
plt.legend()
plt.tight_layout()
plt.savefig("learning_rate_params.png", dpi=300)
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

plt.xticks([d['dataset_size'] for d in data], rotation=45)
plt.xlabel("Dataset size")
plt.ylabel("Metric value")
plt.legend()
plt.tight_layout()
plt.savefig("dataset_size_new_params.png", dpi=300)
plt.close()



















runs_to_plot = [
    "dino-v1-7231276",
    "dino-v1-7388240",
    "dino-v1-7265898",
    "dino-v1-7388214",
    "dino-v1-7364343",
    "dino-v1-7265900",
    "dino-v1-7388224",
    "dino-v1-7234609",
    "dino-v1-7388176",
    "dino-v1-7388189",
]    

path = "./trained_models/LUMI/zip scrapings (huge)/"
paths = [os.path.join(path, run, "summary_metrics.json") for run in runs_to_plot]

data = []
for path in paths:
    with open(path, "r") as f:
        data.append(json.load(f))

for d in data:
    d["alpha"] = np.round((1 - d["momentum"]) * d["dataset_size"] * 256 / (768640 * 2), 5)
    d["cbir_accuracy"] -= 0.2
    print(d["dataset_size"], d["momentum"], d["alpha"])

linestyles = ['-', '--', ':']
cmap = {96080: 'tab:blue', 192160: 'tab:orange', 384320: 'tab:green'}

plt.figure(figsize=(10, 5))

for i, metric in enumerate(['log_loss', 'cbir_accuracy', 'nn_f1']):
    x = np.array([d['alpha'] for d in data])
    y = np.array([d[metric] for d in data])
    z = np.array([d['dataset_size'] for d in data])
    lr = [d['learning_rate'] for d in data]
    zlr = np.array(z) + np.array(lr)
    
    plt.plot([], [], label=labels[metric], color='black', linestyle=linestyles[i])
    
    for z_val in np.unique(zlr):
        idx = np.where(zlr == z_val)[0]
        plt.plot(x[idx], y[idx], marker='o', linestyle=linestyles[i], color=cmap[int(z_val)])
        if i == 2:
            plt.plot([], [], label=f"Dataset size: {z_val}", color=cmap[int(z_val)])

plt.xlabel("Alpha")
plt.ylabel("Metric value")
plt.legend()
plt.tight_layout()
plt.savefig("alpha_params_large_batch.png", dpi=300)
plt.close()
