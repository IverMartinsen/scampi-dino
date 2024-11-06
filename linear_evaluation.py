import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torchvision import datasets
from torchvision import transforms as pth_transforms

import utils
import vision_transformer as vits

from sklearn.neighbors import KNeighborsClassifier  
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    log_loss,
    precision_score,
    log_loss,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    auc,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from lora import LoRA_ViT_timm


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Linear evaluation')
    parser.add_argument('--batch_size', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument('--nb_knn', default=[1, 3, 5, 7, 9], nargs='+', type=int, help='Number of NN to use.')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--data_path', default='/Users/ima029/Desktop/SCAMPI/Repository/data/NO 6407-6-5/labelled imagefolders/imagefolder_20', type=str, help='Path to evaluation dataset')
    parser.add_argument('--lora_rank', default=None, type=int, help='Rank of LoRA projection matrix')
    parser.add_argument('--destination', default='', type=str, help='Destination folder for saving results')
    parser.add_argument('--img_size', default=224, type=int, help='The size of the images used for training the model')
    parser.add_argument('--img_size_pred', default=224, type=int, help='The size of the images used for prediction')
    args = parser.parse_args()

    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    
    # ============ preparing data ... ============
    resize = {96: (110, 110), 224: (256, 256)}
    
    transform = pth_transforms.Compose([
        pth_transforms.Resize(resize[args.img_size_pred], interpolation=3),
        pth_transforms.CenterCrop(args.img_size_pred),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    ds = datasets.ImageFolder(args.data_path, transform=transform)
    
    data_loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    
    # ============ building network ... ============
    print("Building network...")
    model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0, img_size=[args.img_size])
    print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    
    if args.lora_rank is not None:
        model = LoRA_ViT_timm(model, r=args.lora_rank)
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    model.eval()
        
    # ============ extract features ... ============
    print("Extracting features...")

    features = []
    labels = []

    for samples, labs in data_loader:
        features.append(model(samples).detach().numpy())
        labels.append(labs.detach().numpy())
    
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    filenames = [os.path.basename(f[0]) for f in ds.imgs]
    class_names = ds.classes
    
    os.makedirs(args.destination, exist_ok=True)
    
    # compute all pairwise distances
    print("Computing pairwise distances...")
    from sklearn.metrics import pairwise_distances
    dists = pairwise_distances(features, features)
    # add to the diagonal to make the sorting easier
    dists += np.eye(len(labels)) * 1e12
    
    def compute_recall_at_k(labels, dists, k='k'):
        prec_at_k = np.zeros(len(labels))
        rec_at_k = np.zeros(len(labels))

        for i in range(len(labels)):
            if k == 'k':
                _k = np.sum(labels == labels[i]) - 1
            else:
                _k = k
            # get the indices of the k nearest neighbors
            idx = np.argsort(dists[i])[:_k]
            # get the labels of the k nearest neighbors
            nn_labels = labels[idx]
            # count the number of relevant retrieved samples
            n_relevant_retrieved = np.sum(nn_labels == labels[i])
            # count the number of relevant samples
            n_relevant = np.sum(labels == labels[i]) - 1
            # compute the precision at k
            prec_at_k[i] =  n_relevant_retrieved / _k
            # compute the recall at k
            rec_at_k[i] = n_relevant_retrieved / n_relevant
        return prec_at_k, rec_at_k
    
    
    cbir_df = pd.DataFrame()
    cbir_mean_df = pd.DataFrame()
    for k in ['k', 1] + [i for i in range(10, 200, 10)] + [500]:
        prec_at_k, rec_at_k = compute_recall_at_k(labels, dists, k)
        cbir_df[f"precision_at_{k}"] = prec_at_k
        cbir_df[f"recall_at_{k}"] = rec_at_k
        cbir_mean_df.loc["precision", k] = np.mean(prec_at_k)
        cbir_mean_df.loc["recall", k] = np.mean(rec_at_k)

    # plot the precision-recall curve
    x = cbir_mean_df.loc["recall"].values
    y = cbir_mean_df.loc["precision"].values
    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]
    x = [0] + list(x) + [1]
    y = [1] + list(y) + [0]
    
    area = auc(x, y)
    
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, label=f"Area under the curve: {area:.2f}", marker="o", linestyle="--", linewidth=2)
    plt.xlabel("Recall", fontsize=15)
    plt.ylabel("Precision", fontsize=15)
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(args.destination, "precision_recall_curve_cbir.pdf"))
    plt.close()

    
    # compute the mean accuracy for all samples
    
    cbir_df.to_csv(os.path.join(args.destination, "cbir_accuracy.csv"))
    cbir_mean_df.to_csv(os.path.join(args.destination, "cbir_mean_accuracy.csv"))
    
    print("Features are ready!\nStart the classification.")
    
    for label in np.unique(labels):
        print(f"Class {label} has {np.sum(labels == label)} samples in the data set.")

    summary_table = pd.DataFrame()
    summary_tables_knn = {k: pd.DataFrame() for k in args.nb_knn}
    conf_mat_stats = {'preds': [], 'labels': []}
    
    for seed in range(10):
        print(f"Evaluating seed {seed}...")
        
        train_idx, test_idx = train_test_split(np.arange(len(labels)), test_size=0.2, stratify=labels, random_state=seed)
        
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        
        # ============ logistic regression ... ============
        log_model = LogisticRegression(
            max_iter=10000,
            multi_class="multinomial",
            class_weight="balanced",
            random_state=seed,
        )

        log_model.fit(X_train, y_train)
        
        y_pred = log_model.predict(X_test)
        y_proba = log_model.predict_proba(X_test)
        
        conf_mat_stats['preds'].append(y_pred)
        conf_mat_stats['labels'].append(y_test)
            
        summary_table.loc[f"log_model_{seed}", "log_loss"] = log_loss(y_test, y_proba)
        summary_table.loc[f"log_model_{seed}", "balanced_accuracy"] = balanced_accuracy_score(y_test, y_pred)
        summary_table.loc[f"log_model_{seed}", "accuracy"] = accuracy_score(y_test, y_pred)
        summary_table.loc[f"log_model_{seed}", "mean_precision"] = precision_score(y_test, y_pred, average="macro")
        
        # ============ k-NN ... ============
        for k in args.nb_knn:
            print(f"Evaluating k={k}...")
            
            knn = KNeighborsClassifier(n_neighbors=k, p=2)
            
            knn.fit(X_train, y_train)
            
            y_pred = knn.predict(X_test)
        
            # compute the accuracy and balanced accuracy and mean precision on the test set
            summary_tables_knn[k].loc[f"k={k}_seed={seed}", "balanced_accuracy"] = balanced_accuracy_score(y_test, y_pred)
            summary_tables_knn[k].loc[f"k={k}_seed={seed}", "accuracy"] = accuracy_score(y_test, y_pred)
            summary_tables_knn[k].loc[f"k={k}_seed={seed}", "mean_precision"] = precision_score(y_test, y_pred, average="macro")
    
    # ============ summary ... ============
    cm = confusion_matrix(np.concatenate(conf_mat_stats['labels']), np.concatenate(conf_mat_stats['preds']))
    cm_display = ConfusionMatrixDisplay(cm, display_labels=class_names)
    cm_display.plot(xticks_rotation="vertical", colorbar=False, text_kw={"fontsize": 7})
    cm_display.figure_.savefig(os.path.join(args.destination, "confusion_matrix_" + "log_model" + ".pdf"), bbox_inches="tight", dpi=300)

    summary_table.loc["mean", :] = summary_table.mean()
    
    for k in args.nb_knn:
        summary_table_knn = summary_tables_knn[k]
        summary_table_knn.loc["mean", :] = summary_table_knn.mean() 
        summary_table_knn.to_csv(os.path.join(args.destination, f"summary_metrics_knn_{k}.csv"))
    
    summary_table.to_csv(os.path.join(args.destination, "summary_metrics.csv"))

    print("Summary metrics saved.")
