import os
import torch
import utils
import argparse
import numpy as np
import pandas as pd
import vision_transformer as vits
from tqdm import tqdm
from numpy.linalg import svd
from torchvision import datasets
from torchvision import transforms as pth_transforms
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from cbir_utils import compute_recall_at_k, plot_precision_recall_curve, compute_recall_at_k_from_ensemble
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    pairwise_distances,
    f1_score,
)

def compute_recall_at_k_from_ensemble(labels, dists, k='k'):
    prec_at_k = np.zeros(len(labels))
    rec_at_k = np.zeros(len(labels))

    for i in range(len(labels)):
        if k == 'k':
            _k = np.sum(labels == labels[i]) - 1
        else:
            _k = k
        
        # get the indices of the k nearest neighbors
        idx0 = np.argsort(dists[0][i])[:_k]
        idx1 = np.argsort(dists[1][i])[:_k]
        #idx2 = np.argsort(dists[2][i])[:_k]
        
        idx = np.stack([idx0, idx1], axis=1)
        # find the most common index
        idx = np.array([np.argmax(np.bincount(i)) for i in idx])
        
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


def average_weights(pretrained_weights):
    """
    pretrained_weights: list of paths to .pth files
    """
    state_dicts = [torch.load(p, map_location="cpu") for p in pretrained_weights]

    # add all the teacher weights to the first state_dict
    for state_dict in state_dicts[1:]:
        teacher = state_dict['teacher']
        for key in teacher.keys():
            state_dicts[0]['teacher'][key] += teacher[key]

    # divide by the number of state_dicts
    for key in state_dicts[0]['teacher'].keys():
        state_dicts[0]['teacher'][key] /= len(state_dicts)
    
    return state_dicts[0]


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Linear evaluation')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch-size')
    parser.add_argument('--nb_knn', default=[1], nargs='+', type=int, help='List the number of NN to use.')
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--data_path', default='/Users/ima029/Desktop/NO 6407-6-5/data/labelled imagefolders/imagefolder_20', type=str, help='Path to evaluation dataset')
    parser.add_argument('--seed', default=1234, type=int, help='Random seed')
    parser.add_argument('--destination', default='', type=str, help='Destination folder to save results')
    parser.add_argument('--img_size', default=224, type=int)
    parser.add_argument('--merge_mode', default='', type=str, help='weights, features, distances, or outputs')
    parser.add_argument('--normalize_features', default=False, type=bool)
    parser.add_argument('--align_features', default=False, type=bool)
    parser.add_argument('--normalize_distances', default=False, type=bool)
    args = parser.parse_args()
    
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
    ensemble = [
        '/Users/ima029/Desktop/dino-v1/dino/trained_models/LUMI/zip scrapings (huge)/dino-v1-8485178/checkpoint0200.pth',
        '/Users/ima029/Desktop/dino-v1/dino/trained_models/LUMI/zip scrapings (huge)/dino-v1-8485178/checkpoint0250.pth',
        '/Users/ima029/Desktop/dino-v1/dino/trained_models/LUMI/zip scrapings (huge)/dino-v1-8485178/checkpoint.pth',
    ]

    os.makedirs(args.destination, exist_ok=True)
    
    with open(os.path.join(args.destination, "ensemble.txt"), "w") as f:
        for e in ensemble:
            f.write(e + "\n")
    
    if args.merge_mode == "weights":
        model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0, img_size=[args.img_size])
        state_dict = average_weights(ensemble)
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print(f'Average weights loaded with msg: {msg}')
        model.eval()
        models = [model]
    else:
        models = [vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0, img_size=[args.img_size]) for _ in range(len(ensemble))]
        for model, pretrained_weights in zip(models, ensemble):
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
    
    if args.normalize_features:
        features = [f - np.mean(f, axis=0) for f in features]
        features = [f / np.std(f, axis=0) for f in features]
    
    if args.align_features and args.merge_mode != "weights":
        
        reference = 1
        
        for i, f in enumerate(features):
            if i == reference:
                continue
            R = np.dot(features[reference].T, features[i])
        
            U, S, Vh = svd(R)
            
            features[i] = features[i] @ Vh.T @ U.T
    
    if args.merge_mode == "features":
        features = [np.mean(features, axis=0)]
    
    # ============ CBIR evaluation ... ============
    print("CBIR evaluation...")

    dists = [pairwise_distances(f) for f in features]
    if args.normalize_distances:
        dists = [d - np.mean(d, axis=0).reshape(len(labels), 1) for d in dists]
        dists = [d / np.std(d, axis=0).reshape(len(labels), 1) for d in dists]
        dists -= np.min(dists)
    
    dists = [d + np.eye(len(labels)) * 1e12 for d in dists]

    if args.merge_mode == "weights" or args.merge_mode == "features":
        dists = dists[0]
    if args.merge_mode == "distances":
        dists = np.mean(dists, axis=0)
    
    cbir_mean_df = pd.DataFrame()
    
    for k in ['k', 1] + [i for i in range(10, 200, 10)] + [500]:
        if args.merge_mode == "weights" or args.merge_mode == "features" or args.merge_mode == "distances":
            prec_at_k, rec_at_k = compute_recall_at_k(labels, dists, k)
        elif args.merge_mode == "outputs":
            prec_at_k, rec_at_k = compute_recall_at_k_from_ensemble(labels, dists, k)
        cbir_mean_df.loc["precision", k] = np.mean(prec_at_k)
        cbir_mean_df.loc["recall", k] = np.mean(rec_at_k)

    # plot the precision-recall curve
    x = cbir_mean_df.loc["recall"].values
    y = cbir_mean_df.loc["precision"].values
    
    fname = os.path.join(args.destination, "precision_recall_curve_cbir.pdf")
    
    plot_precision_recall_curve(x, y, fname)
    
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
            
            if args.merge_mode == "weights" or args.merge_mode == "features" or args.merge_mode == "distances":
                knn = KNeighborsClassifier(n_neighbors=k, p=2, metric="precomputed")
                knn.fit(dists[train_idx][:, train_idx], labels[train_idx])
                y_pred = knn.predict(dists[test_idx][:, train_idx])

            elif args.merge_mode == "outputs":
            
                preds = []
                
                for i, dist in enumerate(dists):
                    knn = KNeighborsClassifier(n_neighbors=k, p=2, metric="precomputed")
                    knn.fit(dist[train_idx][:, train_idx], labels[train_idx])
                
                    y_pred = knn.predict_proba(dist[test_idx][:, train_idx])
                    
                    preds.append(y_pred)
                
                y_pred = np.stack(preds, axis=1)
                y_pred = np.mean(y_pred, axis=1)
                y_pred = np.argmax(y_pred, axis=1)
            
            summary_tables_knn[k].loc[f"k={k}_seed={seed}", "balanced_accuracy"] = balanced_accuracy_score(y_test, y_pred)
            summary_tables_knn[k].loc[f"k={k}_seed={seed}", "accuracy"] = accuracy_score(y_test, y_pred)
            summary_tables_knn[k].loc[f"k={k}_seed={seed}", "mean_precision"] = precision_score(y_test, y_pred, average="macro")
            summary_tables_knn[k].loc[f"k={k}_seed={seed}", "f1_score"] = f1_score(y_test, y_pred, average="macro")

    for k in args.nb_knn:
        summary_table_knn = summary_tables_knn[k]
        summary_table_knn.loc["mean", :] = summary_table_knn.mean() 
        summary_table_knn.to_csv(os.path.join(args.destination, f"summary_metrics_knn_{k}.csv"))
    
