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
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Linear evaluation')
    parser.add_argument('--batch_size', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument('--nb_knn', default=[1, 3, 5, 7, 9], nargs='+', type=int, help='Number of NN to use.')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--data_path', default='./labelled crops 20', type=str, help='Path to evaluation dataset')
    parser.add_argument('--lora_rank', default=None, type=int, help='Rank of LoRA projection matrix')
    parser.add_argument('--seed', default=1234, type=int, help='Random seed')
    parser.add_argument('--destination', default='', type=str, help='Destination folder to save results')
    parser.add_argument('--img_size', default=224, type=int)
    args = parser.parse_args()

    args.pretrained_weights = "/Users/ima029/Desktop/SCAMPI/Repository/scampi_unsupervised/frameworks/dino/trained_models/vit_small_fine_tuned/checkpoint0002.pth"
    args.data_path = "/Users/ima029/Desktop/SCAMPI/Repository/data/NO 6407-6-5/labelled imagefolders/imagefolder_20_classes"
    
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    
    # ============ preparing data ... ============
    transform = pth_transforms.Compose([
        pth_transforms.Resize((256, 256), interpolation=3),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    dataset_train = datasets.ImageFolder(os.path.join(args.data_path, "train"), transform=transform)
    dataset_val = datasets.ImageFolder(os.path.join(args.data_path, "val"), transform=transform)
    
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
    )

    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")

    # ============ building network ... ============
    model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0, img_size=[args.img_size])
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    model.eval()
    print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    
    # ============ extract features ... ============
    print("Extracting features for train set...")

    train_features = []
    train_labels = []

    for samples, labels in data_loader_train:
        train_features.append(model(samples).detach().numpy())
        train_labels.append(labels.detach().numpy())
        
    train_features = np.concatenate(train_features, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)
    train_filenames = [os.path.basename(f[0]) for f in dataset_train.imgs]

    print("Extracting features for val set...")
    
    test_features = []
    test_labels = []

    for samples, labels in data_loader_val:
        test_features.append(model(samples).detach().numpy())
        test_labels.append(labels.detach().numpy())

    test_features = np.concatenate(test_features, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)
    test_filenames = [os.path.basename(f[0]) for f in dataset_val.imgs]
    
    
    for label in np.unique(test_labels):
        print(f"Class {label} has {np.sum(test_labels == label)} samples in the test set.")
    
    print("Features are ready!\nStart the classification.")
    
    os.makedirs(args.destination, exist_ok=True)
    
    class_names = dataset_train.classes
    # table for class wise metrics
    table = pd.DataFrame()
    table["class"] = class_names
    # summary table for overall metrics
    summary_table = pd.DataFrame()
    
    # ============ k-NN ... ============

    k = 1
    knn = KNeighborsClassifier(n_neighbors=k, p=2)
    knn.fit(train_features, train_labels)
    y_pred = knn.predict(test_features)
    
    # compute the accuracy and balanced accuracy and mean precision on the test set
    summary_table.loc[f"k={k}", "balanced_accuracy"] = balanced_accuracy_score(test_labels, y_pred)
    summary_table.loc[f"k={k}", "accuracy"] = accuracy_score(test_labels, y_pred)
    summary_table.loc[f"k={k}", "mean_precision"] = precision_score(test_labels, y_pred, average="macro")
    
    # add the metrics to the table
    table[f"k={k}" + "_precision"] = precision_score(test_labels, y_pred, average=None)
    table[f"k={k}" + "_recall"] = recall_score(test_labels, y_pred, average=None)
    table[f"k={k}" + "_f1_score"] = f1_score(test_labels, y_pred, average=None)


    # save confusion matrix
    cm = confusion_matrix(test_labels, y_pred)
    cm_display = ConfusionMatrixDisplay(
        cm, display_labels=class_names
        )
    cm_display.plot(xticks_rotation="vertical", colorbar=False)
    cm_display.figure_.show()
    #cm_display.figure_.savefig(os.path.join(args.destination, "confusion_matrix_" + "log_model" + ".pdf"), bbox_inches="tight", dpi=300)
    
    # also save predictions for each image
    preds = pd.DataFrame()
    preds["filename"] = test_filenames
    preds["true_label"] = test_labels
    preds["pred_label"] = y_pred
    preds.to_csv(os.path.join(args.destination, "predictions_" + "log_model" + ".csv"))
        
    # save pandas table as csv to same folder as pretrained weights
    table.to_csv(os.path.join(args.destination, "class_wise_metrics.csv"))
    summary_table.to_csv(os.path.join(args.destination, "summary_metrics.csv"))




# standardize test_features to have zero mean and unit variance
test_features = (test_features - np.mean(test_features, axis=0)) / np.std(test_features, axis=0)


good: [6, 7, 17]
bad: [1, 9, 10, 12, 13, 15, 16]
from PIL import Image
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt

idxs = np.where((test_labels == 1) | (test_labels == 15))[0]
#idxs = np.arange(64, 96)




filenames = [os.path.join(args.data_path, "val", class_names[test_labels[i]],  test_filenames[i]) for i in idxs]
features = (test_features[idxs] > 0) * 1.0
labs = test_labels[idxs]
pred = y_pred[idxs]

A = np.matmul(features, features.transpose()) # shape (num_patches, num_patches)
#A = distance_matrix(features, features, p=2)
## find three nearest neighbors
#A = np.zeros((len(features), len(features)))
#D = distance_matrix(features, features, p=2)
#for i in range(A.shape[0]):
#    _idxs = np.argsort(D[i])[:3]
#    A[i, _idxs] = 1

#plt.imshow(B)
#plt.show()

#A = A * B
#A = np.max(A) - A
#A = np.exp(-A**2 / 350)

# normalize
#A = A - np.min(A)
#A = A / np.max(A)
#A = np.exp(A)
#A += 1e-3
A = A / np.sum(A, axis=(0))
#A = A.T



for i in range(20):
    A = np.matmul(A, A)

rankings = A[0]


fig, axs = plt.subplots(6, 4, figsize=(20, 5))

axes = axs.flatten()

for i, ax in enumerate(axes):
    j = np.argsort(rankings)[::-1][i]
    img = Image.open(filenames[j])
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(f"Rank: {rankings[j]:.2f}, True: {labs[j]}, Pred: {pred[j]}", fontsize=6)
plt.show()



