import tensorflow as tf
import torch
from torch.utils.data import Dataset, IterableDataset


class TFRecordMapDataset(Dataset):
    def __init__(self, tfrecord_files, feature_description, transform=None):
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
        self.dataset = tf.data.TFRecordDataset(self.tfrecord_files)
        self.transform = transform  # Add any other arguments you need
        
    def _parse_function(self, example_proto):
        # Parse the input `tf.train.Example` proto using the feature description.
        return tf.io.parse_single_example(example_proto, self.feature_description)
    
    def __len__(self):
        # This is not very efficient as it goes through the whole dataset.
        # Consider caching the length if it doesn't change.
        return sum(1 for _ in self.dataset)
    
    def __getitem__(self, idx):
        # Since TFRecord datasets do not support direct indexing, we need to iterate.
        # For large datasets, consider using a more efficient approach.
        for i, raw_record in enumerate(self.dataset):
            if i == idx:
                example = self._parse_function(raw_record)
                # Convert the example to a format suitable for your task, e.g., PyTorch tensors
                # This example assumes a single feature called 'image' which is an image tensor
                image = example['image'].numpy()  # Convert to numpy array
                image_tensor = torch.from_numpy(image).float()  # Convert to PyTorch tensor
                if self.transform:
                    image_tensor = self.transform(image_tensor)
                return image_tensor
        raise IndexError("Index out of range")


class TFRecordIterableDataset(IterableDataset):
    def __init__(self, tfrecord_files, feature_description, transform=None):
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
        self.dataset = tf.data.TFRecordDataset(self.tfrecord_files)
        self.transform = transform  # Add any other arguments you need

    def parse_example(self, example_proto):
        # Parse the input `tf.train.Example` proto using the dictionary.
        return tf.io.parse_single_example(example_proto, self.feature_description)
    
    def generator(self):
        for raw_record in self.dataset:
            example = self.parse_example(raw_record)
            # Process the example (convert to tensors, etc.)
            # Assuming an image stored as bytes
            image = tf.io.decode_image(example['image'])
            image_tensor = torch.from_numpy(image.numpy()).float()
            if self.transform:
                image_tensor = self.transform(image_tensor)
            yield image_tensor

    def __iter__(self):
        return iter(self.generator())
