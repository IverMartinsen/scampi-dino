import torch
from torchvision import datasets
from torchvision import transforms as pth_transforms
from vision_transformer import vit_small
from utils import load_pretrained_weights


# load the model
model = vit_small(patch_size=16, num_classes=0, img_size=[224])

load_pretrained_weights(model, "path/to/weights", "teacher", "vit_small", 16)

model.eval()


# load the data
transform = pth_transforms.Compose([
    pth_transforms.Resize((256, 256), interpolation=3),
    pth_transforms.CenterCrop(224),
    pth_transforms.ToTensor(),
    pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
    
dataset = datasets.ImageFolder("path/to/data", transform=transform)

data_loader = torch.utils.data.DataLoader(dataset, batch_size=128)


# extract features
for samples, labels in data_loader:
    features = model(samples).detach().numpy()
    labels = labels.detach().numpy()
