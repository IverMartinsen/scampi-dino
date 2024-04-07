import os
import argparse

import numpy as np
import pandas as pd

import torch
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
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.utils import compute_class_weight
from lora import LoRA_ViT_timm


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Linear evaluation')
    parser.add_argument('--batch_size', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument('--nb_knn', default=[1, 3, 5, 7, 9], nargs='+', type=int, help='Number of NN to use.')
    #parser.add_argument('--pretrained_weights', default='./trained_models/vit_small_fine_tuned/checkpoint0002.pth', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--data_path', default='/Users/ima029/Desktop/SCAMPI/Repository/data/NO 6407-6-5/labelled imagefolders/imagefolder_20_classes_not_split', type=str, help='Path to evaluation dataset')
    parser.add_argument('--lora_rank', default=None, type=int, help='Rank of LoRA projection matrix')
    parser.add_argument('--seed', default=1234, type=int, help='Random seed')
    parser.add_argument('--destination', default='', type=str, help='Destination folder to save results')
    parser.add_argument('--img_size', default=224, type=int)
    parser.add_argument('--img_size_pred', default=224, type=int)
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
    
    pretrained_ensemble_weights = [
        "./trained_models/vit_small_fine_tuned/old/checkpoint0000.pth",
        "./trained_models/vit_small_fine_tuned/old/checkpoint0001.pth",
        "./trained_models/vit_small_fine_tuned/old/checkpoint0002.pth",
        "./trained_models/vit_small_fine_tuned/old/checkpoint0003.pth",
        "./trained_models/vit_small_fine_tuned/old/checkpoint0004.pth",
        #"./trained_models/vit_small_fine_tuned/old/checkpoint0005.pth",
        #"./trained_models/vit_small_fine_tuned/old/checkpoint0006.pth",
        #"./trained_models/vit_small_fine_tuned/old/checkpoint0007.pth",
        #"./trained_models/vit_small_fine_tuned/old/checkpoint0008.pth",
        #"./trained_models/vit_small_fine_tuned/old/checkpoint0009.pth",
    ]
    
    
    # ============ building network ... ============
    models = [vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0, img_size=[args.img_size]) for _ in range(len(pretrained_ensemble_weights))]
        
    for model, pretrained_weights in zip(models, pretrained_ensemble_weights):
        utils.load_pretrained_weights(model, pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
        model.eval()
    
    print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    
    # ============ extract features ... ============
    print("Extracting features...")

    features = [[] for _ in range(len(models))]
    labels = []

    for samples, labs in data_loader:
        labels.append(labs.detach().numpy())
        for i, model in enumerate(models):
            features[i].append(model(samples).detach().numpy())
        
    features = [np.concatenate(f, axis=0) for f in features]
    labels = np.concatenate(labels, axis=0)
    filenames = [os.path.basename(f[0]) for f in ds.imgs]

    
    os.makedirs(args.destination, exist_ok=True)

    summary_table = pd.DataFrame()
    summary_table_knn = pd.DataFrame()

    for seed in range(10):
    
        # create a train and test split
        train_idx, test_idx = train_test_split(np.arange(len(labels)), test_size=0.2, stratify=labels, random_state=seed)
        
        test_labels = labels[test_idx]
        
        #for label in np.unique(test_labels):
        #    print(f"Class {label} has {np.sum(test_labels == label)} samples in the test set.")
        
        print("Features are ready!\nStart the classification.")
        
        
        class_names = ds.classes
        # table for class wise metrics
        #table = pd.DataFrame()
        #table["class"] = class_names
        # summary table for overall metrics
        
        probs = []
        
        # ============ logistic regression ... ============
        for i, (f, model) in enumerate(zip(features, models)):
            log_model = LogisticRegression(
                max_iter=10000,
                multi_class="multinomial",
                class_weight="balanced",
                random_state=seed,
            )
            
            log_model.fit(f[train_idx], labels[train_idx])
            probs.append(log_model.predict_proba(f[test_idx]))
            
        probs = np.stack(probs, axis=2)
        y_proba = np.mean(probs, axis=2)
        y_pred = np.argmax(y_proba, axis=1)
            
        summary_table.loc[f"log_model_{seed}", "log_loss"] = log_loss(test_labels, y_proba)
        summary_table.loc[f"log_model_{seed}", "balanced_accuracy"] = balanced_accuracy_score(test_labels, y_pred)
        summary_table.loc[f"log_model_{seed}", "accuracy"] = accuracy_score(test_labels, y_pred)
        summary_table.loc[f"log_model_{seed}", "mean_precision"] = precision_score(test_labels, y_pred, average="macro")
        
        
        ## save confusion matrix
        #cm = confusion_matrix(test_labels, y_pred)
        #cm_display = ConfusionMatrixDisplay(
        #    cm, display_labels=class_names
        #    )
        #cm_display.plot(xticks_rotation="vertical", colorbar=False)
        #cm_display.figure_.savefig(os.path.join(args.destination, "confusion_matrix_" + "log_model" + ".pdf"), bbox_inches="tight", dpi=300)
        
        # ============ k-NN ... ============
        #for k in args.nb_knn:
        k = 7
        #knn = KNeighborsClassifier(n_neighbors=k, p=2)
        #
        # compute distance between features
        from sklearn.metrics.pairwise import euclidean_distances
        #
        distances = [euclidean_distances(f) for f in features]
        distances = np.stack(distances, axis=2)
        distances = np.mean(distances, axis=2)
        
            
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
        knn = KNeighborsClassifier(n_neighbors=k, metric="precomputed")
        knn.fit(distances[train_idx][:, train_idx], labels[train_idx])
        y_pred = knn.predict(distances[test_idx][:, train_idx])

        summary_table_knn.loc[f"k={k}_{seed}", "balanced_accuracy"] = balanced_accuracy_score(test_labels, y_pred)
        summary_table_knn.loc[f"k={k}_{seed}", "accuracy"] = accuracy_score(test_labels, y_pred)
        summary_table_knn.loc[f"k={k}_{seed}", "mean_precision"] = precision_score(test_labels, y_pred, average="macro")

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
    summary_table.loc["mean", :] = summary_table.mean()
    
    summary_table_knn.loc["mean", :] = summary_table_knn.mean()
    
    summary_table.to_csv(os.path.join(args.destination, "summary_metrics.csv"))

    summary_table_knn.to_csv(os.path.join(args.destination, "summary_metrics_knn.csv"))
