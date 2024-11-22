import os
import argparse

import numpy as np
import pandas as pd

import torch
from tqdm import tqdm
from torchvision import datasets
from torchvision import transforms as pth_transforms

import utils
import vision_transformer as vits

from sklearn.neighbors import KNeighborsClassifier  
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    log_loss,
    precision_score,
    log_loss,
    pairwise_distances,
    f1_score,
)
from sklearn.linear_model import LogisticRegression
from cbir_utils import compute_recall_at_k, plot_precision_recall_curve
from numpy.linalg import svd


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Linear evaluation')
    parser.add_argument('--batch_size', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument('--nb_knn', default=[1, 3, 5, 7, 9], nargs='+', type=int, help='Number of NN to use.')
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--data_path', default='/Users/ima029/Desktop/SCAMPI/Repository/data/NO 6407-6-5/labelled imagefolders/imagefolder_20', type=str, help='Path to evaluation dataset')
    parser.add_argument('--lora_rank', default=None, type=int, help='Rank of LoRA projection matrix')
    parser.add_argument('--seed', default=1234, type=int, help='Random seed')
    parser.add_argument('--destination', default='', type=str, help='Destination folder to save results')
    parser.add_argument('--img_size', default=224, type=int)
    parser.add_argument('--img_size_pred', default=224, type=int)
    args = parser.parse_args()

    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    
    pretrained_ensemble_weights = [
        '/Users/ima029/Desktop/dino-v1/dino/trained_models/LUMI/zip scrapings (huge)/dino-v1-8370959/checkpoint.pth',
        '/Users/ima029/Desktop/dino-v1/dino/trained_models/LUMI/zip scrapings (huge)/dino-v1-8370959/checkpoint0260.pth',
        #'/Users/ima029/Desktop/dino-v1/dino/trained_models/LUMI/zip scrapings (huge)/dino-v1-7268043/checkpoint.pth',
        #'/Users/ima029/Desktop/dino-v1/dino/trained_models/LUMI/zip scrapings (huge)/dino-v1-8370995/checkpoint.pth',
    ]

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
    models = [vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0, img_size=[args.img_size]) for _ in range(len(pretrained_ensemble_weights))]
        
    for model, pretrained_weights in zip(models, pretrained_ensemble_weights):
        utils.load_pretrained_weights(model, pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
        model.eval()
    
    # ============ extract features ... ============
    print("Extracting features...")

    features = [[] for _ in range(len(models))]
    labels = []

    for _, (samples, labs) in enumerate(tqdm(data_loader)):
        labels.append(labs.detach().numpy())
        for i, model in enumerate(models):
            features[i].append(model(samples).detach().numpy())
        
    features = [np.concatenate(f, axis=0) for f in features]
    labels = np.concatenate(labels, axis=0)
    filenames = [os.path.basename(f[0]) for f in ds.imgs]
    
    normalize_features = True
    if normalize_features:
        features = [f - np.mean(f, axis=0) for f in features]
        features = [f / np.std(f, axis=0) for f in features]

    align_features = True
    
    if align_features:
        
        reference = 1
        
        for i, f in enumerate(features):
            if i == reference:
                continue
            R = np.dot(features[reference].T, features[i])
        
            U, S, Vh = svd(R)
            
            features[i] = features[i] @ Vh.T @ U.T

    average_features = True
    
    if average_features:
        features = np.mean(features, axis=0)

    os.makedirs(args.destination, exist_ok=True)

    # ============ CBIR evaluation ... ============
    print("CBIR evaluation...")

    dists = [pairwise_distances(f) for f in features]
    
    normalize_distances = True
    if normalize_distances:
        dists = [d - np.mean(d, axis=0) for d in dists]
        dists = [d / np.std(d, axis=0) for d in dists]
        #dists = [d + np.eye(len(labels)) * 1e12 for d in dists]

    average_distances = True
    if average_distances:
        dists = np.mean(dists, axis=0)
    
    dists -= np.min(dists)
    
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
    
    fname = os.path.join(args.destination, "precision_recall_curve_cbir.pdf")
    
    plot_precision_recall_curve(x, y, fname)
    
    cbir_df.to_csv(os.path.join(args.destination, "cbir_accuracy.csv"))
    cbir_mean_df.to_csv(os.path.join(args.destination, "cbir_mean_accuracy.csv"))

    print("CBIR evaluation done.")

    summary_table = pd.DataFrame()
    summary_tables_knn = {k: pd.DataFrame() for k in args.nb_knn}

    
    
    for seed in range(10):
    
        # create a train and test split
        train_idx, test_idx = train_test_split(np.arange(len(labels)), test_size=0.2, stratify=labels, random_state=seed)
        
        #X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        # ============ k-NN ... ============
        for k in args.nb_knn:
            print(f"Evaluating k={k}...")
            
            #from sklearn.neighbors import NearestNeighbors
            
            #knn = NearestNeighbors(n_neighbors=k, p=2, metric="precomputed")
            
            preds = []
            
            for i, dist in enumerate(dists):
                knn = KNeighborsClassifier(n_neighbors=k, p=2, metric="precomputed")
                knn.fit(dist[train_idx][:, train_idx], labels[train_idx])
            
                y_pred = knn.predict_proba(dist[test_idx][:, train_idx])
                
                #y_pred = knn.kneighbors(dist[test_idx][:, train_idx])[1]
                #y_pred = labels[train_idx][y_pred]
                #y_pred = np.array([np.argmax(np.bincount(i)) for i in y_pred])
                preds.append(y_pred)
            
            y_pred = np.stack(preds, axis=1)
            y_pred = np.mean(y_pred, axis=1)
            y_pred = np.argmax(y_pred, axis=1)
            
            #knn.fit(X_train, y_train)
            
            #y_pred = knn.predict(X_test)
        
            # compute the accuracy and balanced accuracy and mean precision on the test set
            summary_tables_knn[k].loc[f"k={k}_seed={seed}", "balanced_accuracy"] = balanced_accuracy_score(y_test, y_pred)
            summary_tables_knn[k].loc[f"k={k}_seed={seed}", "accuracy"] = accuracy_score(y_test, y_pred)
            summary_tables_knn[k].loc[f"k={k}_seed={seed}", "mean_precision"] = precision_score(y_test, y_pred, average="macro")
            summary_tables_knn[k].loc[f"k={k}_seed={seed}", "f1_score"] = f1_score(y_test, y_pred, average="macro")

        #test_labels = labels[test_idx]
        
        #for label in np.unique(test_labels):
        #    print(f"Class {label} has {np.sum(test_labels == label)} samples in the test set.")
        
        #print("Features are ready!\nStart the classification.")
        
        
        #class_names = ds.classes
        # table for class wise metrics
        #table = pd.DataFrame()
        #table["class"] = class_names
        # summary table for overall metrics
        
        #probs = []
        #
        ## ============ logistic regression ... ============
        #for i, (f, model) in enumerate(zip(features, models)):
        #    log_model = LogisticRegression(
        #        max_iter=10000,
        #        multi_class="multinomial",
        #        class_weight="balanced",
        #        random_state=seed,
        #    )
        #    
        #    log_model.fit(f[train_idx], labels[train_idx])
        #    probs.append(log_model.predict_proba(f[test_idx]))
        #    
        #probs = np.stack(probs, axis=2)
        #y_proba = np.mean(probs, axis=2)
        #y_pred = np.argmax(y_proba, axis=1)
        #    
        #summary_table.loc[f"log_model_{seed}", "log_loss"] = log_loss(test_labels, y_proba)
        #summary_table.loc[f"log_model_{seed}", "balanced_accuracy"] = balanced_accuracy_score(test_labels, y_pred)
        #summary_table.loc[f"log_model_{seed}", "accuracy"] = accuracy_score(test_labels, y_pred)
        #summary_table.loc[f"log_model_{seed}", "mean_precision"] = precision_score(test_labels, y_pred, average="macro")
        
        
        ## save confusion matrix
        #cm = confusion_matrix(test_labels, y_pred)
        #cm_display = ConfusionMatrixDisplay(
        #    cm, display_labels=class_names
        #    )
        #cm_display.plot(xticks_rotation="vertical", colorbar=False)
        #cm_display.figure_.savefig(os.path.join(args.destination, "confusion_matrix_" + "log_model" + ".pdf"), bbox_inches="tight", dpi=300)
        
        # ============ k-NN ... ============
        #for k in args.nb_knn:
        

        
        #k = 7
        ##knn = KNeighborsClassifier(n_neighbors=k, p=2)
        ##
        ## compute distance between features
        #from sklearn.metrics.pairwise import euclidean_distances
        ##
        #distances = [euclidean_distances(f) for f in features]
        #distances = np.stack(distances, axis=2)
        #distances = np.mean(distances, axis=2)
        
            
        #distances = euclidean_distances(features)
        #distances_pretrained = euclidean_distances(pretrained_features)
        #distances_combined = (distances + distances_pretrained) / 2
        #
        #knn.fit(features[train_idx], labels[train_idx])
        #y_pred = knn.predict(features[test_idx])
        #
        ## compute the accuracy and balanced accuracy and mean precision on the test set
        #
        #knn.fit(pretrained_features[train_idx], labels[train_idx])
        #y_pred_pretrained = knn.predict(pretrained_features[test_idx])
        
        
        
        
        # compute the accuracy and balanced accuracy and mean precision on the test set
        #summary_table_knn_pretrained.loc[f"k={k}_{seed}", "balanced_accuracy"] = balanced_accuracy_score(test_labels, y_pred_pretrained)
        #summary_table_knn_pretrained.loc[f"k={k}_{seed}", "accuracy"] = accuracy_score(test_labels, y_pred_pretrained)
        #summary_table_knn_pretrained.loc[f"k={k}_{seed}", "mean_precision"] = precision_score(test_labels, y_pred_pretrained, average="macro")
        #
        #knn = KNeighborsClassifier(n_neighbors=k, metric="precomputed")
        #knn.fit(distances[train_idx][:, train_idx], labels[train_idx])
        #y_pred = knn.predict(distances[test_idx][:, train_idx])
#
        #summary_table_knn.loc[f"k={k}_{seed}", "balanced_accuracy"] = balanced_accuracy_score(test_labels, y_pred)
        #summary_table_knn.loc[f"k={k}_{seed}", "accuracy"] = accuracy_score(test_labels, y_pred)
        #summary_table_knn.loc[f"k={k}_{seed}", "mean_precision"] = precision_score(test_labels, y_pred, average="macro")

        #y_pred_combined = knn.predict(distances_combined[test_idx][:, train_idx])
#
        #summary_table_knn_combined.loc[f"k={k}_{seed}", "balanced_accuracy"] = balanced_accuracy_score(test_labels, y_pred_combined)
        #summary_table_knn_combined.loc[f"k={k}_{seed}", "accuracy"] = accuracy_score(test_labels, y_pred_combined)
        #summary_table_knn_combined.loc[f"k={k}_{seed}", "mean_precision"] = precision_score(test_labels, y_pred_combined, average="macro")
        
        # add the metrics to the table
        #table[f"k={k}" + "_precision"] = precision_score(test_labels, y_pred, average=None)
        #table[f"k={k}" + "_recall"] = recall_score(test_labels, y_pred, average=None)
        #table[f"k={k}" + "_f1_score"] = f1_score(test_labels, y_pred, average=None)
            
        # save pandas table as csv to same folder as pretrained weights
        #table.to_csv(os.path.join(args.destination, "class_wise_metrics.csv"))
    # compute the mean
    #summary_table.loc["mean", :] = summary_table.mean()
    
    #summary_table_knn.loc["mean", :] = summary_table_knn.mean()
    
    #summary_table.to_csv(os.path.join(args.destination, "summary_metrics.csv"))

    #summary_table_knn.to_csv(os.path.join(args.destination, "summary_metrics_knn.csv"))

    for k in args.nb_knn:
        summary_table_knn = summary_tables_knn[k]
        summary_table_knn.loc["mean", :] = summary_table_knn.mean() 
        summary_table_knn.to_csv(os.path.join(args.destination, f"summary_metrics_knn_{k}.csv"))
    
