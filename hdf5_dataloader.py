import os
import h5py
import torch
import torchvision
from PIL import Image
from typing import List, Dict, Tuple
from torchvision import datasets, transforms

def load_image_from_hdf5(hdf5_file, key):
    with h5py.File(hdf5_file, 'r') as f:
        bytes = f[key][()]
        image = torch.tensor(bytes) # sequence of bytes
        image = torchvision.io.decode_jpeg(image) # shape: (3, H, W)
        image = image.permute(1, 2, 0) # shape: (H, W, 3)
        image = image.numpy()
        image = Image.fromarray(image)
    return image

def make_dataset(hdf5_file):
    with h5py.File(hdf5_file, 'r') as f:
        keys = list(f.keys())
    return list(zip(keys, [0]*len(keys)))

def make_group_dataset(root):
    files = [f for f in os.listdir(root) if f.endswith('.hdf5')]
    files = [os.path.join(root, f) for f in files]
    samples = []
    for f in files:
        with h5py.File(f, 'r') as h:
            keys = list(h.keys())
        n = len(keys)
        keys = zip([f]*n, keys)
        samples += list(zip(keys, [0]*n))
    return samples

class HDF5Dataset(datasets.DatasetFolder):
    def __init__(self, root, loader=load_image_from_hdf5, transform=None, *args, **kwargs):
        super().__init__(root, loader, transform, *args, **kwargs)
        
        samples = make_dataset(root)
        self.transform = transform
        self.samples = samples
        self.classes = [0]
        self.class_to_idx = {0: 0} 
    
    @staticmethod
    def make_dataset(path_to_file, *args, **kwargs):
        return make_dataset(path_to_file)
    
    def __getitem__(self, idx):
        key, target = self.samples[idx]
        sample = self.loader(self.root, key)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target
        
    def __len__(self):
        return len(self.samples)
    
    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        return ['0'], {'0': 0}


class HDF5GroupDataset(datasets.DatasetFolder):
    def __init__(self, root, loader=load_image_from_hdf5, transform=None, *args, **kwargs):
        super().__init__(root, loader, transform, *args, **kwargs)
        
        samples = make_group_dataset(root)
        self.transform = transform
        self.samples = samples
        self.classes = [0]
        self.class_to_idx = {0: 0} 
    
    @staticmethod
    def make_dataset(path_to_file, *args, **kwargs):
        return make_group_dataset(path_to_file)
    
    def __getitem__(self, idx):
        (file, key), target = self.samples[idx]
        sample = self.loader(file, key)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target
        
    def __len__(self):
        return len(self.samples)
    
    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        return ['0'], {'0': 0}


if __name__ == "__main__":
    path = "/Users/ima029/Desktop/SCAMPI/Repository/data/zip scrapings (huge)/run_2024-02-08-hdf5/shard_1.hdf5"

    class DataAugmentation(object):
        def __init__(self, img_size=224):
            self.img_size = img_size

        def __call__(self, image):
            # resize image to img_size
            image = transforms.functional.resize(image, (self.img_size, self.img_size))
            return image

    transform = DataAugmentation()
    dataset = HDF5Dataset(path, transform=transform)
    print(dataset.transform)
    print(len(dataset))
    image, label = dataset[0]
    print(f"Image type: {type(image)}")
    print(label)
    print(dataset.classes)
    print(dataset.class_to_idx)
    print(dataset.find_classes(path))
    
    path_to_group = "/Users/ima029/Desktop/SCAMPI/Repository/data/zip scrapings (huge)/run_2024-02-08-hdf5"
    group_dataset = HDF5GroupDataset(path_to_group, transform=transform)
    print(len(group_dataset))
    image, label = group_dataset[0]
    print(f"Image type: {type(image)}")
    print(group_dataset.transform)