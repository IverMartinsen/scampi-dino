import os
import torch
import utils
import argparse
import numpy as np
import matplotlib.pyplot as plt
import vision_transformer as vits
from torchvision import datasets
from torchvision import transforms as pth_transforms


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Linear evaluation')
    parser.add_argument('--batch_size', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--data_path', default='/Users/ima029/Desktop/NO 6407-6-5/data/labelled imagefolders/imagefolder_20', type=str, help='Path to evaluation dataset')
    parser.add_argument('--destination', default='', type=str, help='Destination folder for saving results')
    args = parser.parse_args()
    
    # prepare data
    transform = pth_transforms.Compose([
        pth_transforms.Resize((256, 256), interpolation=3),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    ds = datasets.ImageFolder(args.data_path, transform=transform)
    
    data_loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    
    args.pretrained_weights = '/Users/ima029/Desktop/dino-v1/dino/trained_models/LUMI/zip scrapings (huge)/dino-v1-8485178/checkpoint.pth'
    args.pretrained_weights = ''
    # vit base
    args.pretrained_weights = '/Users/ima029/Desktop/dino-v1/dino/trained_models/LUMI/zip scrapings (huge)/dino-v1-8590394/checkpoint.pth'
    args.arch = 'vit_base'
    
    # ============ building network ... ============
    print("Building network...")
    model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0, img_size=[224])
    print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    
    model.eval()
    
    # Extract features
    features = []
    labels = []

    for samples, labs in data_loader:
        features.append(model(samples).detach().numpy())
        labels.append(labs.detach().numpy())
    
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    filenames = [os.path.basename(f[0]) for f in ds.imgs]
    class_names = ds.classes
    
    for label in np.unique(labels):
        print(f"Class {label} has {np.sum(labels == label)} samples in the data set.")
    
    os.makedirs(args.destination, exist_ok=True)
    
    # ANOVA
    x_mean = np.mean(features, axis=0)
    k = len(np.unique(labels))
    N = len(labels)
    SSA = 0
    for i in range(k):
        idx = np.where(labels == i)[0]
        n = np.sum(labels == i)
        SSA += n * ((x_mean - np.mean(features[idx], axis=0))**2)
    
    SST = np.sum((features - x_mean)**2, axis=0)
    SSE = SST - SSA
    
    ratio = SSA / SSE

    sigma_sq_b = SSA / (k - 1)
    sigma_sq_w = SSE / (N - k)

    f_ratio = sigma_sq_b / sigma_sq_w
    
    print(f"Number of features with SS_b/SS_w > 1: {np.sum(ratio > 1)}")

    plt.figure()
    plt.bar(np.arange(1, features.shape[1] + 1), ratio)
    plt.xlabel("Features")
    plt.ylabel(r"$\frac{SS_b}{SS_w}$")
    plt.axhline(1, color="black", linestyle="--")
    plt.xticks([1,] + list(range(100, features.shape[1] + 1, 100)) + [features.shape[1]])
    plt.ylim(0, 3)
    plt.savefig(os.path.join(args.destination, "ssb_ssw.pdf"), bbox_inches="tight", dpi=300)
    plt.close()
