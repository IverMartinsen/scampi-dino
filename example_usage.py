import torch
from torchvision import datasets, transforms
from vision_transformer import vit_small, vit_base
from utils import load_pretrained_weights

path_to_data = "path/to/data"
path_to_weights = "path/to/weights"

if __name__ == "__main__":

    # load the model
    model = vit_small(patch_size=16, num_classes=0, img_size=[224])

    load_pretrained_weights(model, path_to_weights, "teacher", "vit_small", 16)

    model.eval()

    # preprocess the data
    transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    # load the dataset
    dataset = datasets.ImageFolder(path_to_data, transform=transform)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=128)

    # extract features
    for samples, labels in data_loader:
        features = model(samples).detach().numpy()
        labels = labels.detach().numpy()


