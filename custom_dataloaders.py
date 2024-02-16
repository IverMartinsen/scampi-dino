import glob
import h5py
import tensorflow as tf
import torch
from PIL import Image
from torch.utils.data import Dataset, IterableDataset


class TFRecordMapDataset(Dataset):
    def __init__(self, tfrecord_files, feature_description={"image": tf.io.FixedLenFeature([], tf.string)}, transform=None, length=None):
        """
        Args:
        - tfrecord_files (list of str): List of paths to TFRecord files.
        - feature_description (dict): A dictionary describing the features in the TFRecord file.
        
        # Example usage:
        
        # Define your TFRecord feature description
        feature_description = {
            'image': tf.io.FixedLenFeature([], tf.string),
            # Add other features here
        }

        # Paths to your TFRecord files
        tfrecord_files = ['path/to/tfrecord1', 'path/to/tfrecord2']

        # Create the dataset
        dataset = TFRecordMapDataset(tfrecord_files, feature_description)

        # Use DataLoader to load the data
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        """
        self.tfrecord_files = tfrecord_files
        self.feature_description = feature_description
        self.dataset = tf.data.TFRecordDataset(glob.glob(self.tfrecord_files + "*.tfrecords"))
        self.transform = transform  # Add any other arguments you need
        if length is not None:
            self.length = length
        else:
            self.length = sum(1 for _ in self.dataset)
        
    def _parse_function(self, example_proto):
        # Parse the input `tf.train.Example` proto using the feature description.
        x = tf.io.parse_single_example(example_proto, self.feature_description)
        x = tf.io.parse_tensor(x["image"], out_type=tf.uint8)
        return x
    
    def __len__(self):
        # This is not very efficient as it goes through the whole dataset.
        # Consider caching the length if it doesn't change.
        #return sum(1 for _ in self.dataset)
        return self.length
    
    def __getitem__(self, idx):
        # Since TFRecord datasets do not support direct indexing, we need to iterate.
        # For large datasets, consider using a more efficient approach.
        for i, raw_record in enumerate(self.dataset):
            if i == idx:
                example = self._parse_function(raw_record)
                # Convert the example to a format suitable for your task, e.g., PyTorch tensors
                # This example assumes a single feature called 'image' which is an image tensor
                image = example.numpy()
                image_tensor = torch.from_numpy(image).float()  # Convert to PyTorch tensor
                if self.transform:
                    image_tensor = self.transform(image_tensor)
                return image_tensor
        raise IndexError("Index out of range")


class TFRecordIterableDataset(IterableDataset):
    def __init__(self, tfrecord_files, feature_description={"image": tf.io.FixedLenFeature([], tf.string)}, transform=None, length=None):
        """
        Example usage:

        # Define your TFRecord feature description
        feature_description = {
            'image': tf.io.FixedLenFeature([], tf.string),
            # Add other features here
        }

        # Paths to your TFRecord files
        tfrecord_files = ['path/to/tfrecord1', 'path/to/tfrecord2']

        # Create the dataset
        iterable_dataset = TFRecordIterableDataset(tfrecord_files, feature_description)

        # Use DataLoader to load the data
        # Note: Since this is an IterableDataset, shuffle and sampler options are not applicable.
        dataloader = DataLoader(iterable_dataset, batch_size=32)
        """
        self.tfrecord_files = tfrecord_files
        self.feature_description = feature_description
        self.dataset = tf.data.TFRecordDataset(glob.glob(self.tfrecord_files + "*.tfrecords"))
        self.transform = transform  # Add any other arguments you need
        self.length = length
    
    def __len__(self):
        if self.length is not None:
            return self.length
        else:
            return sum(1 for _ in self.dataset)

    def parse_example(self, example_proto):
        # Parse the input `tf.train.Example` proto using the dictionary.
        x = tf.io.parse_single_example(example_proto, self.feature_description)
        x = tf.io.parse_tensor(x["image"], out_type=tf.uint8)

        return x
    
    def generator(self):
        for raw_record in self.dataset:
            example = self.parse_example(raw_record)
            # Process the example (convert to tensors, etc.)
            # Assuming an image stored as bytes
            image = example.numpy()
            image_tensor = torch.from_numpy(image.numpy()).float()
            if self.transform:
                image_tensor = self.transform(image_tensor)
            yield image_tensor

    def __iter__(self):
        return iter(self.generator())


class HDF5Dataset(Dataset):
    def __init__(self, hdf5_file, transform=None, *args, **kwargs):
        super(HDF5Dataset, self).__init__(*args, **kwargs)
        self.hdf5_file = hdf5_file
        self.transform = transform

    def __len__(self):
        with h5py.File(self.hdf5_file, 'r') as h5f:
            return len(h5f['images'])

    def __getitem__(self, idx):
        with h5py.File(self.hdf5_file, 'r') as h5f:
            image = h5f['images'][idx]
        # Note that the DINO dataloader expects a PIL image
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        # Note that the DINO dataloader expects a tuple (PIL.Image, label)
        return image, 0
