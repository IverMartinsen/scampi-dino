import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import vision_transformer as vits
import utils

from sklearn.manifold import TSNE
from torchvision import transforms as pth_transforms
from PIL import Image

df = pd.read_csv('/Users/ima029/Desktop/SCAMPI/Repository/data/NO 6407-6-5/labelled crops/labels.csv')

filenames = df['filename'].values
firstnames = df['firstname'].values
lastnames = df['lastname'].values
labels = [f"{f}_{l}" for f, l in zip(firstnames, lastnames)]

transform = pth_transforms.Compose([
    pth_transforms.Resize([256, 256], interpolation=3),
    pth_transforms.CenterCrop(224),
    pth_transforms.ToTensor(),
    pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, filenames, labels, transform, path):
        self.filenames = filenames
        self.labels = labels
        self.transform = transform
        self.path = path
        
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img = Image.open(self.path + self.filenames[idx])
        img = self.transform(img)
        return img, self.labels[idx]

ds = ImageDataset(filenames, labels, transform, path='/Users/ima029/Desktop/SCAMPI/Repository/data/NO 6407-6-5/labelled crops/')

data_loader = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=False)

model = vits.__dict__['vit_small'](patch_size=16, num_classes=0, img_size=[224])
utils.load_pretrained_weights(model, '/Users/ima029/Desktop/dino-v1/dino/trained_models/scampi/6407-6-5/fine tuning/vit_small_fine_tuned_256_1e5/checkpoint0005.pth', 'teacher', 'vit_small', 16)
model.eval()

feats = []
labs = []

for samples, labs_ in data_loader:
    feats.append(model(samples).detach().numpy())
    labs.append(labs_)

feats = np.concatenate(feats, axis=0)
labs = np.concatenate(labs, axis=0)




t_sne = TSNE(n_components=2, random_state=0)
X_2d = t_sne.fit_transform(feats)
x = X_2d[:,0]
y = X_2d[:,1]

selected_classes = [
    'alisocysta_margarita', 
    'azolla_massulae', 
    'eatonicysta_ursulae', 
    'inaperturopollenites_hiatus', 
    'isabelidinium_cooksoniae',
    'isabelidinium_viborgense', 
    'svalbardella_cooksoniae',
    'svalbardella_kareniae',
    'svalbardella_clausii',
    ]

#colors = ['red', 'green', 'blue', 'yellow', 'orange', 'purple']

# custom colormap
import matplotlib.colors as mcolors
tab20 = plt.cm.tab20.colors
tab20c = plt.cm.tab20c.colors
colors = (tab20[0], tab20[16], tab20[10], tab20[8], tab20c[4], tab20c[5], tab20c[8], tab20c[9], tab20c[10])
cmap = mcolors.ListedColormap(colors)


plt.figure(figsize=(12,8))
plt.scatter(x, y, c='black', alpha=0.2)
for i, c in enumerate(selected_classes):
    idx = np.where(labs == c)
    # use a cross
    plt.scatter(x[idx], y[idx], label=c, alpha=0.8, c=colors[i], s=50)
plt.legend()
plt.axis('off')
plt.savefig('t_sne.pdf', bbox_inches='tight', pad_inches=0, dpi=300)
#plt.show()

