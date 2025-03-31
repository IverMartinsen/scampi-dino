import os
import json
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torchvision import datasets
from torchvision import transforms as pth_transforms


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
    f1_score,
    pairwise_distances
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from cbir_utils import compute_recall_at_k, plot_precision_recall_curve, retrieve_filenames
from eval_utils import load_vit_mae_model, load_dinov2_model, load_dino_model


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Linear evaluation')
    parser.add_argument('--batch_size', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument('--nb_knn', default=[1, 3, 5, 7, 9], nargs='+', type=int, help='Number of NN to use.')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--data_path', default='/Users/ima029/Desktop/NO 6407-6-5/data/labelled imagefolders/imagefolder_20', type=str, help='Path to evaluation dataset')
    parser.add_argument('--lora_rank', default=None, type=int, help='Rank of LoRA projection matrix')
    parser.add_argument('--destination', default='', type=str, help='Destination folder for saving results')
    args = parser.parse_args()

    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    
    # ============ preparing data ... ============    
    transform = pth_transforms.Compose([
        pth_transforms.Resize((256, 256), interpolation=3),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    ds = datasets.ImageFolder(args.data_path, transform=transform)
    
    data_loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    
    # ============ building network ... ============
    if args.pretrained_weights == 'vit_mae':
        model = load_vit_mae_model(args)
    elif args.pretrained_weights == 'dinov2':
        model = load_dinov2_model(args)
    else:
        model = load_dino_model(args)
    
    model.eval()
    print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    
    raise
    
    # get model attributes
    #try:
    #    checkpoint = torch.load(args.pretrained_weights, map_location="cpu")
    #
    #    train_args = {}
    #    for a in vars(checkpoint["args"]):
    #        train_args[a] = getattr(checkpoint["args"], a)
    #    
    #    summary_json = {}
    #    summary_json["batch_size"] = train_args["batch_size_per_gpu"] * train_args["world_size"]
    #    summary_json["learning_rate"] = train_args["lr"]
    #    summary_json["momentum"] = train_args["momentum_teacher"]
    #    summary_json["num_epochs"] = train_args["epochs"]
    #    
    #    files_used = train_args["data_path"].split("/")[5]
    #    print(f"Files used: {files_used}")
    #    if files_used.endswith("test"):
    #        summary_json["dataset_size"] = 3074560
    #    elif files_used.endswith("16"):
    #        summary_json["dataset_size"] = 1537280
    #    elif files_used.endswith("8"):
    #        summary_json["dataset_size"] = 768640
    #    elif files_used.endswith("4"):
    #        summary_json["dataset_size"] = 384320
    #    elif files_used.endswith("2"):
    #        summary_json["dataset_size"] = 192160
    #    elif files_used.endswith("1"):
    #        summary_json["dataset_size"] = 96080
    #    else:
    #        summary_json["dataset_size"] = 0
    #except FileNotFoundError:
    #    print("Checkpoint not found.")

    # ============ extract features ... ============
    print("Extracting features...")

    features = []
    labels = []

    for samples, labs in data_loader:
        if args.pretrained_weights == 'vit_mae':
            y = model.forward_features(samples)
        else:
            y = model(samples)
        features.append(y.detach().numpy())
        labels.append(labs.detach().numpy())
    
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    filenames = [os.path.basename(f[0]) for f in ds.imgs]
    class_names = ds.classes

    print("Features are ready!\nStart the classification.")
    
    for label in np.unique(labels):
        print(f"Class {label} has {np.sum(labels == label)} samples in the data set.")
    
    os.makedirs(args.destination, exist_ok=True)
    
    print("CBIR evaluation...")
    
    dists = pairwise_distances(features, features)
    dists += np.eye(len(labels)) * 1e12
    
    cbir_df = pd.DataFrame({"label": labels, "filename": filenames})
    cbir_mean_df = pd.DataFrame()
    for k in ['k', 1] + [i for i in range(10, 200, 10)] + [500]:
        prec_at_k, rec_at_k = compute_recall_at_k(labels, dists, k)
        cbir_df[f"precision_at_{k}"] = prec_at_k
        cbir_df[f"recall_at_{k}"] = rec_at_k
        cbir_mean_df.loc["precision", k] = np.mean(prec_at_k)
        cbir_mean_df.loc["recall", k] = np.mean(rec_at_k)

    # filenames of the top5 and bot5
    top10 = cbir_df["filename"][cbir_df["precision_at_k"].argsort()[-10:][::-1]]
    bot10 = cbir_df["filename"][cbir_df["precision_at_k"].argsort()[:10]]

    import matplotlib.pyplot as plt
    from PIL import Image
    
    def get_cname(fname):
        lab = labels[np.where(np.array(filenames) == fname)[0][0]]
        return class_names[lab]
    
    def get_fpath(fname):
        return os.path.join(args.data_path, get_cname(fname), fname)
    
    def get_precision_at_k(fname):
        return cbir_df[cbir_df["filename"] == fname][f"precision_at_k"].values[0]
    
    fig, axes = plt.subplots(10, 10, figsize=(20, 20))
    
    for i in range(10):
        query = top10.iloc[i]
        axes[i, 0].imshow(Image.open(get_fpath(query)).resize((224, 224)))
        axes[i, 0].set_title(get_cname(query) + " (query)" + f"\n(precision@k={get_precision_at_k(query):.2f})")
        axes[i, 0].axis("off")
        
        retrieved_filenames = retrieve_filenames(query, labels, filenames, dists)
        
        for j in range(1, 10):
            axes[i, j].imshow(Image.open(get_fpath(retrieved_filenames[j])).resize((224, 224)))
            axes[i, j].set_title(get_cname(retrieved_filenames[j]))
            axes[i, j].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(args.destination, "top10_retrieved_images.pdf"), bbox_inches="tight", dpi=300)
    plt.close()

    fig, axes = plt.subplots(10, 10, figsize=(20, 20))
    
    for i in range(10):
        query = bot10.iloc[i]
        axes[i, 0].imshow(Image.open(get_fpath(query)).resize((224, 224)))
        axes[i, 0].set_title(get_cname(query) + " (query)" + f"\n(precision@k={get_precision_at_k(query):.2f})")
        axes[i, 0].axis("off")
        
        retrieved_filenames = retrieve_filenames(query, labels, filenames, dists)
        
        for j in range(1, 10):
            axes[i, j].imshow(Image.open(get_fpath(retrieved_filenames[j])).resize((224, 224)))
            axes[i, j].set_title(get_cname(retrieved_filenames[j]))
            axes[i, j].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(args.destination, "bot10_retrieved_images.pdf"), bbox_inches="tight", dpi=300)
    plt.close()


    # compute mean for each class
    pd.DataFrame({
        "class": np.unique(labels),
        "precision_at_k": [cbir_df["precision_at_k"][cbir_df["label"] == k].mean() for k in np.unique(labels)]},
    ).to_csv(os.path.join(args.destination, "cbir_classwise.csv"))
    
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
    class_wise_nn_stats = []
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
            summary_tables_knn[k].loc[f"k={k}_seed={seed}", "f1_score"] = f1_score(y_test, y_pred, average="macro")

            # class wise recall, precision and f1 score
            if k == 1:
                df = pd.DataFrame({
                    "recall": recall_score(y_test, y_pred, average=None),
                    "precision": precision_score(y_test, y_pred, average=None),
                    "f1_score": f1_score(y_test, y_pred, average=None),
                })
                class_wise_nn_stats.append(df)
            
        # ============ OOD detection ... ============
        for k in np.unique(labels):
            X_train_in, X_test_in = X_train[y_train != k], X_test[y_test != k]
            X_test_out = np.concatenate((X_test[y_test == k], X_train[y_train == k]), axis=0)
            y_test_in = np.ones(len(X_test_in))
            y_test_out = np.zeros(len(X_test_out))
            
            X_test_ood = np.concatenate([X_test_in, X_test_out], axis=0)
            y_test_ood = np.concatenate([y_test_in, y_test_out], axis=0)
            
            #log_model = LogisticRegression(
            #    max_iter=1000,
            #    multi_class="multinomial",
            #    class_weight="balanced",
            #    random_state=seed,
            #)
            #log_model.fit(X_train_in, y_train[y_train != k])
            #ood_proba = log_model.predict_proba(X_test_ood)
            #ood_entropy = -np.sum(ood_proba * np.log(ood_proba + 1e-12), axis=1)
            
            
            ood_dists = pairwise_distances(X_test_ood, X_train_in)
            
            # MDS visualization
            #from sklearn.manifold import MDS
            #mds = MDS(n_components=2, random_state=seed)
            #mds.fit(np.concatenate([X_train_in, X_test_ood], axis=0))
            #xy = mds.embedding_
            #plt.figure(figsize=(10, 5))
            #plt.scatter(xy[:len(X_train_in), 0], xy[:len(X_train_in), 1], label="Train", alpha=0.5)
            #plt.scatter(xy[len(X_train_in):, 0], xy[len(X_train_in):, 1], label="Test", alpha=0.5, c=y_test_ood)
            #plt.legend()
            #plt.tight_layout()
            #plt.show()
            
            
            ood_dists = ood_dists.mean(axis=1)
            
            #plt.figure(figsize=(10, 5))
            #plt.hist(ood_dists[y_test_ood == 0], bins=10, alpha=0.5, label="OOD", density=True)
            #plt.hist(ood_dists[y_test_ood == 1], bins=10, alpha=0.5, label="IN", density=True)
            #plt.legend()
            #plt.xlabel("Mean distance to IN samples")
            #plt.ylabel("Number of samples")
            #plt.title(f"OOD detection for class {class_names[k]}")
            #plt.tight_layout()
            #plt.show()
        
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

    # take the mean of the class wise statistics
    class_wise_nn_stats_mean = pd.concat(class_wise_nn_stats).groupby(level=0).mean()
    class_wise_nn_stats_mean.to_csv(os.path.join(args.destination, "class_wise_nn_stats.csv"))
    
    try:
        summary_json["log_loss"] = summary_table.loc["mean", "log_loss"]
        summary_json["cbir_accuracy"] = cbir_mean_df.loc["precision", "k"]
        summary_json["nn_f1"] = summary_tables_knn[1].mean()["f1_score"]
        
        with open(os.path.join(args.destination, "summary_metrics.json"), "w") as f:
            json.dump(summary_json, f)
    except NameError:
        print("Json not saved.")
    
    print("Summary metrics saved.")
