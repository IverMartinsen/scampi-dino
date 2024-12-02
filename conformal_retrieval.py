import os
import torch
import utils
import argparse
import numpy as np
import pandas as pd
import vision_transformer as vits
from torchvision import datasets
from torchvision import transforms as pth_transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances, cosine_distances


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Linear evaluation')
    parser.add_argument('--batch_size', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--data_path', default='/Users/ima029/Desktop/NO 6407-6-5/labelled imagefolders/imagefolder_20', type=str, help='Path to evaluation dataset')
    parser.add_argument('--destination', default='', type=str, help='Destination folder for saving results')
    parser.add_argument('--alpha', default=0.05, type=float, help='Significance level')
    args = parser.parse_args()

    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    
    # ============ preparing data ... ============
    resize = {96: (110, 110), 224: (256, 256)}
        
    transform = pth_transforms.Compose([
        pth_transforms.Resize(resize[224], interpolation=3),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    ds = datasets.ImageFolder(args.data_path, transform=transform)
    
    data_loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    
    # ============ building network ... ============
    print("Building network...")
    if args.pretrained_weights == 'dinov2':
        torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    else:
        model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0, img_size=[224])
        print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
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

    print("Features are ready!\nStart the classification.")
    
    for label in np.unique(labels):
        print(f"Class {label} has {np.sum(labels == label)} samples in the data set.")
    
    os.makedirs(args.destination, exist_ok=True)
    
    print("CBIR evaluation...")
    
    dataframes = []
    
    for seed in range(10):

        print(f"Evaluating seed {seed}...")
        
        cal_idx, val_idx = train_test_split(np.arange(len(labels)), test_size=0.5, stratify=labels, random_state=seed)
        
        X_cal, X_val = features[cal_idx], features[val_idx]
        y_cal, y_val = labels[cal_idx], labels[val_idx]
        
        args.alpha = 0.15
        
        # compute the similarity matrix
        A = cosine_similarity(X_cal, X_cal)
        
        Y = np.zeros_like(A)
        for k in np.unique(y_cal):
            idx = np.dot((y_cal == k).reshape(-1, 1), (y_cal == k).reshape(1, -1))
            Y[idx] = 1
        Y = Y - np.eye(len(y_cal))
        
        
        def get_adaptive_scores_unlabeled(x):
            A = cosine_distances(x, x) + np.eye(len(x)) * 1000
            simranks = A.argsort(1)[:, :]            
            cumsum_by_rank = np.take_along_axis(A, simranks, axis=1).cumsum(1)
            scores = np.take_along_axis(cumsum_by_rank, simranks.argsort(1), axis=1)
            return scores
        
        def get_adaptive_scores(x, y):
            
            A = cosine_distances(x, x) + np.eye(len(x)) * 1000
            
            Y = np.zeros_like(A)
            for k in np.unique(y):
                idx = np.dot((y == k).reshape(-1, 1), (y == k).reshape(1, -1))
                Y[idx] = 1
            Y = Y - np.eye(len(y))
            
            simranks = A.argsort(1)[:, :]
            
            cumsum_by_rank = np.take_along_axis(A, simranks, axis=1).cumsum(1)
            
            nn_labs = np.argmin(A * Y + (1 - Y) * 1000, axis=1)
            
            S = np.take_along_axis(cumsum_by_rank, simranks.argsort(1), axis=1)[np.arange(len(A)), nn_labs]
            return S
        
        
        A_reduced = np.max(A * Y, axis=1)
        
        S = 1 - A_reduced
        
        S = get_adaptive_scores(X_cal[np.where(y_cal == 11)[0]], y_cal[y_cal == 11])
        
        q_level = np.ceil((1 - args.alpha)*(len(S) + 1)) / len(S)
        
        q = np.quantile(S, q_level)
        
        threshold = 1 - q
        
        A_val = cosine_similarity(X_val, X_val) - np.eye(len(X_val))
        
        A_val = get_adaptive_scores_unlabeled(X_val)
        
        #proposals = (A_val > threshold)
        proposals = (A_val < q)
        
        # assert no empty proposals
        proposals[np.arange(len(y_val)), np.argmax(A_val, axis=0)] = True
        
        Y_val = np.zeros_like(A_val)
        for k in np.unique(y_val):
            idx = np.dot((y_val == k).reshape(-1, 1), (y_val == k).reshape(1, -1))
            Y_val[idx] = 1
        Y_val = Y_val - np.eye(len(y_val))
        
        ((proposals * Y_val).sum(axis=1) > 0).mean()
        
        class_wise_metrics = pd.DataFrame({"class": np.unique(labels)})

        for k in np.unique(y_val):
            
            proposals_class_k = proposals[np.where(y_val == k)[0], :]
            num_proposals_class_k = proposals_class_k.sum(axis=1)
            print("====================================")
            print(f"Class {k} has {num_proposals_class_k.mean().round(1)} proposals.")
            class_wise_metrics.loc[class_wise_metrics["class"] == k, f"num_proposals"] = num_proposals_class_k.mean().round(1)
            # correct proposals
            correct_proposals_class_k = (proposals_class_k * Y_val[np.where(y_val == k)[0], :]).sum(axis=1)
            print(f"{correct_proposals_class_k.mean().round(1)} are relevant.")
            class_wise_metrics.loc[class_wise_metrics["class"] == k, f"correct_proposals"] = correct_proposals_class_k.mean().round(1)
            correct_proposals_class_k = correct_proposals_class_k > 0
            print(f"Prediction set has contains correct class {correct_proposals_class_k.mean()}.")
            class_wise_metrics.loc[class_wise_metrics["class"] == k, f"contains_class"] = correct_proposals_class_k.mean()
        
        dataframes.append(class_wise_metrics)
    
    df = pd.concat(dataframes)
    df = df.groupby("class").mean()
    df["contains_class"].mean()