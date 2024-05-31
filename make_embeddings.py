import os
import argparse
import h5py
import torch
import utils
import numpy as np
import vision_transformer as vits
from torchvision import transforms as pth_transforms
from hdf5_dataloader_v2 import HDF5Dataset

if __name__ == '__main__':

    parser = argparse.ArgumentParser('Linear evaluation')
    parser.add_argument('--batch_size', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--data_path', default='/Users/ima029/Desktop/SCAMPI/Repository/data/zip scrapings (huge)/run_2024-02-08-hdf5-test', type=str, help='Path to evaluation dataset')
    parser.add_argument('--destination', default='', type=str, help='Destination folder for saving results')
    args = parser.parse_args()

    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    
    # check if GPU is available
    print("GPU available: ", torch.cuda.is_available())
    
    # ============ preparing data ... ============
    transform = pth_transforms.Compose([
        pth_transforms.Resize((256, 256), interpolation=3),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    files = [os.path.join(args.data_path, f) for f in os.listdir(args.data_path) if f.endswith(".hdf5")]
    
    os.makedirs(args.destination, exist_ok=True)
    
    # ============ building network ... ============
    print("Building network...")
    model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0, img_size=[224])
    print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    model.eval()
    # use the gpu
    model = model.cuda()


    for file in files:
        
        print(f"Processing {file}")
        
        target_file = os.path.basename(file) + "_features.hdf5"
                
        # check if the target file already exists
        if target_file in os.listdir(args.destination):
            print(f"File {target_file} already exists. Skipping...")
            continue
        
        ds = HDF5Dataset(file, transform=transform)
    
        data_loader = torch.utils.data.DataLoader(
            ds, 
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=10,)
    
        # ============ extract features ... ============
        print("Extracting features...")

        hdf5_file = [os.path.basename(f[0][0]) for f in ds.samples]
        filenames = [os.path.basename(f[0][1]) for f in ds.samples]
    
        features = []
    
        for i, (samples, _) in enumerate(data_loader):
            print(f"Batch {i+1}/{len(data_loader)}")
            samples = samples.cuda()
            features.append(model(samples).detach().cpu().numpy())
    
        features = np.concatenate(features, axis=0)
    
        with h5py.File(os.path.join(args.destination, target_file), 'w') as f:
            f.create_dataset("features", data=features)
            f.create_dataset("filenames", data=filenames[:len(features)])
            f.create_dataset("hdf5_file", data=hdf5_file[:len(features)])
