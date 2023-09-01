import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
import os

train_path = "/home/pirenjie/data/coco/annotations/instances_train2017.json"
val_path = "/home/pirenjie/data/coco/annotations/instances_val2017.json"
save_path = "/home/pirenjie/data/coco/masks"
if not os.path.exists(save_path):
    os.makedirs(save_path)
if not os.path.exists(os.path.join(save_path, "train")):
    os.makedirs(os.path.join(save_path, "train"))
if not os.path.exists(os.path.join(save_path, "val")):
    os.makedirs(os.path.join(save_path, "val"))

coco_train = COCO(train_path)
coco_val = COCO(val_path)


annotations_train = coco_train.dataset["annotations"]
for i, anno in enumerate(annotations_train):
    mask_pixel = coco_train.annToMask(anno)
    plt.imsave(os.path.join(save_path, "train", f"mask_{i}.png"), mask_pixel, cmap='gray')
# annotations_val = coco_val.dataset["annotations"]
# for i, anno in enumerate(annotations_val):
#     mask_pixel = coco_val.annToMask(anno)
#     plt.imsave(os.path.join(save_path, "val", f"mask_{i}.png"), mask_pixel, cmap='gray')
#

