from typing import Dict, List, Tuple
from PIL import Image
from zipfile import ZipFile 
from torchvision import datasets

def make_dataset(zip_file):
    instances = []
    with ZipFile(zip_file) as z:
        for f in z.namelist():
            if f.endswith('.jpg'):
                instances.append(f)
    return instances

def load_image_from_zip(zip_file, file_name):
    with ZipFile(zip_file) as z:
        with z.open(file_name) as f:
            image = Image.open(f)
            image = image.resize((224, 224))
    return image


class ZipFilesDataset(datasets.DatasetFolder):
    def __init__(self, root, loader=load_image_from_zip, transform=None, *args, **kwargs):
        super(ZipFilesDataset, self).__init__(root, loader, transform, *args, **kwargs)
        
        samples = make_dataset(root) # This is a list of file paths in the zip file
        self.samples = samples
        self.classes = [0]
        self.class_to_idx = {0: 0} 
        
    @staticmethod
    def make_dataset(path_to_zip, *args, **kwargs):
        return make_dataset(path_to_zip)
    
    def __getitem__(self, idx):
        path = self.samples[idx]
        sample = self.loader(self.root, path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, 0

    def __len__(self):
        return len(self.samples)
    
    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        return ['0'], {'0': 0}

if __name__ == "__main__":
    dataset = ZipFilesDataset("/Users/ima029/Desktop/SCAMPI/Repository/data/zip scrapings (huge)/run_2024-02-08.zip")
    print(len(dataset))
    print(dataset[0])
    print(dataset.classes)
    print(dataset.class_to_idx)
    print(dataset.find_classes("data.zip"))