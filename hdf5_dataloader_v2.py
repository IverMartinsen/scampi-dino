import os
import h5py
import torch
import torchvision
import numpy as np
from torchvision.datasets import VisionDataset
from PIL import Image


def load_image_from_hdf5(path):
    hdf5_file, key = path
    with h5py.File(hdf5_file, 'r') as f:
        bytes = f[key][()]
        image = torch.tensor(bytes) # sequence of bytes
        image = torchvision.io.decode_jpeg(image) # shape: (3, H, W)
        image = image.permute(1, 2, 0) # shape: (H, W, 3)
        image = image.numpy()
        image = Image.fromarray(image)
    return image


def load_png_from_hdf5(path):
    hdf5_file, key = path
    with h5py.File(hdf5_file, 'r') as f:
        bytes = f[key][()]
        image = torch.tensor(bytes) # sequence of bytes
        image = torchvision.io.decode_png(image) # shape: (3, H, W)
        image = image.permute(1, 2, 0) # shape: (H, W, 3)
        image = image.numpy()
        image = Image.fromarray(image)
    return image


def load_jpeg_from_hdf5(path):
    hdf5_file, key = path
    with h5py.File(hdf5_file, 'r') as f:
        bytes = f[key][()]
        image = torch.tensor(bytes)
        image = torchvision.io.decode_jpeg(image)
        image = image.permute(1, 2, 0)
        image = image.numpy()
        image = Image.fromarray(image)
    return image


def make_dataset(
    root,
    class_to_idx,
    extensions=None,
    is_valid_file=None,
    allow_empty=False,
    ):
    with h5py.File(root, 'r') as f:
        keys = list(f.keys())
    return np.array(list(zip(keys, [0]*len(keys))))


def make_group_dataset(
    root,
    class_to_idx,
    extensions=None,
    is_valid_file=None,
    allow_empty=False,
    ):
    files = [f for f in os.listdir(root) if f.endswith('.hdf5')]
    samples = []
    for f in files:
        with h5py.File(os.path.join(root, f), 'r') as h:
            keys = list(h.keys())
        n = len(keys)
        samples += list(zip([f]*n, keys, [0]*n))
    return np.array(samples)


def find_classes(directory):
    classes = [0]
    class_to_idx = {0: 0} 
    return classes, class_to_idx


class HDF5Dataset(VisionDataset):
    def __init__(
        self,
        root,
        loader=load_png_from_hdf5,
        extensions=None,
        transform=None,
        target_transform=None,
        is_valid_file=None,
        allow_empty=False,
    ):
        super().__init__(
            root, 
            transform=transform, 
            target_transform=target_transform
        )
        classes, class_to_idx = self.find_classes(self.root)
        samples = self.make_dataset(
            self.root,
            class_to_idx=class_to_idx,
            extensions=extensions,
            is_valid_file=is_valid_file,
            allow_empty=allow_empty,
        )

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = np.array([s[1] for s in samples])

    @staticmethod
    def make_dataset(
        directory,
        class_to_idx,
        extensions=None,
        is_valid_file=None,
        allow_empty=False,
    ):
        if class_to_idx is None:
            # prevent potential bug since make_dataset() would use the class_to_idx logic of the
            # find_classes() function, instead of using that of the find_classes() method, which
            # is potentially overridden and thus could have a different logic.
            raise ValueError("The class_to_idx parameter cannot be None.")
        return make_dataset(
            directory, class_to_idx, extensions=extensions, is_valid_file=is_valid_file, allow_empty=allow_empty
        )

    def find_classes(self, directory):
        return find_classes(directory)

    def __getitem__(self, index):
        key, target = self.samples[index]
        sample = self.loader((self.root, key))
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)



class HDF5GroupDataset(VisionDataset):
    """A generic data loader.

    This default directory structure can be customized by overriding the
    :meth:`find_classes` method.

    Args:
        root (str or ``pathlib.Path``): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.
        allow_empty(bool, optional): If True, empty folders are considered to be valid classes.
            An error is raised on empty folders if False (default).

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(
        self,
        root,
        loader=load_jpeg_from_hdf5,
        extensions=None,
        transform=None,
        target_transform=None,
        is_valid_file=None,
        allow_empty=False,
    ):
        super().__init__(
            root, 
            transform=transform, 
            target_transform=target_transform
        )
        classes, class_to_idx = self.find_classes(self.root)
        samples = self.make_dataset(
            self.root,
            class_to_idx=class_to_idx,
            extensions=extensions,
            is_valid_file=is_valid_file,
            allow_empty=allow_empty,
        )

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = np.array([s[1] for s in samples])

    @staticmethod
    def make_dataset(
        directory,
        class_to_idx,
        extensions=None,
        is_valid_file=None,
        allow_empty=False,
    ):
        if class_to_idx is None:
            # prevent potential bug since make_dataset() would use the class_to_idx logic of the
            # find_classes() function, instead of using that of the find_classes() method, which
            # is potentially overridden and thus could have a different logic.
            raise ValueError("The class_to_idx parameter cannot be None.")
        return make_group_dataset(
            directory, class_to_idx, extensions=extensions, is_valid_file=is_valid_file, allow_empty=allow_empty
        )

    def find_classes(self, directory):
        return find_classes(directory)

    def __getitem__(self, index):
        filename, key, target = self.samples[index]
        path = os.path.join(self.root, filename)
        sample = self.loader((path, key))
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)


if __name__ == "__main__":
    path_to_group = "/Users/ima029/Desktop/SCAMPI/Repository/data/zip scrapings (huge)/run_2024-02-08-hdf5"
    group_dataset = HDF5GroupDataset(path_to_group)
    print(len(group_dataset))
    image, label = group_dataset[1000000]
    print(f"Image type: {type(image)}")
    print(group_dataset.transform)
    image.show()
    
    # compute RGB mean and std for the dataset
    mean = []
    std = []
    
    idx = np.random.choice(len(group_dataset), 100000)
    
    for i in range(len(group_dataset)):
        img, _ = group_dataset[idx[i]]
        img = img.resize((224, 224))
        img = np.array(img)
        img = img / 255.
        mean.append(np.mean(img, axis=(0,1)))
        std.append(np.std(img, axis=(0,1)))
        if i % 1000 == 0:
            print(f"Processed {i} images")
    
    mean = np.array(mean).mean(axis=0)
    std = np.array(std).mean(axis=0)