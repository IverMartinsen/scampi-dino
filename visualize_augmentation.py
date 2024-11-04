from PIL import Image
from main_dino import DataAugmentationDINO

image = Image.open('/Users/ima029/Desktop/SCAMPI/Repository/data/NO 6407-6-5/labelled imagefolders/imagefolder_20/areoligera/image249.jpg')

global_crops_scale=(0.4, 1.)
local_crops_scale=(0.05, 0.4)
local_crops_number=8

transform = DataAugmentationDINO(global_crops_scale=global_crops_scale, local_crops_scale=local_crops_scale, local_crops_number=local_crops_number)

views = transform(image)

for i, view in enumerate(views):
    view = view.permute(1, 2, 0).numpy()
    view -= view.min()
    view /= view.max()
    view *= 255
    view = view.astype('uint8')
    view = Image.fromarray(view)
    view.save(f'image249_{i}.jpg')
