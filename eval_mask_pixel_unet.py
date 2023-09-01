"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
import os
import time
import torch
from torch import nn, optim
from torch.optim import Adam, AdamW
import numpy as np
from models.unet.unet_model import UNetNoShortCut as UNet
from segmentation_mask_dataset_vae import SegmentationMaskDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import argparse
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from focal_loss import focal_loss
from torch.optim.lr_scheduler import CosineAnnealingLR
from refcoco.refer import REFER
import torchvision.transforms as transforms

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform(m.weight.data)


# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='Plot training and validation losses.')

# Add arguments
parser.add_argument('--figname', type=str, default='losses.jpeg',
                    help='filename for saving the figure (default: losses.jpeg)')
parser.add_argument('--pretrained', type=str, default=None,
                    help='filename for saving the figure (default: losses.jpeg)')
parser.add_argument('--lr', type=float, default=5e-5,
                    help='learning rate (default: 0.001)')
parser.add_argument('--epoch', type=int, default=50,
                    help='learning rate (default: 0.001)')
parser.add_argument('--local-rank', type=int, default=-1,
                    help='learning rate (default: 0.001)')

parser.add_argument('--factor', type=float, default=0.9,
                    help='learning rate (default: 0.001)')
parser.add_argument('--clip', type=float, default=1.0,
                    help='learning rate (default: 0.001)')
parser.add_argument('--adam_eps', type=float, default=5e-9,
                    help='learning rate (default: 0.001)')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='learning rate (default: 0.001)')
parser.add_argument('--patience', type=int, default=10,
                    help='learning rate (default: 0.001)')
parser.add_argument('--warmup', type=int, default=100,
                    help='learning rate (default: 0.001)')
parser.add_argument('--train_size', type=int, default=None,
                    help='learning rate (default: 0.001)')
parser.add_argument('--val_size', type=int, default=None,
                    help='learning rate (default: 0.001)')
parser.add_argument('--train_bs', type=int, default=100,
                    help='learning rate (default: 0.001)')
parser.add_argument('--val_bs', type=int, default=100,
                    help='learning rate (default: 0.001)')
parser.add_argument('--cls_loss', type=str, default="cross_entropy",
                    help='learning rate (default: 0.001)')
parser.add_argument('--cls_weight', type=float, default=1.,
                    help='learning rate (default: 0.001)')
parser.add_argument('--dist_weight', type=float, default=.1,
                    help='learning rate (default: 0.001)')
parser.add_argument('--reg_weight', type=float, default=1.,
                    help='learning rate (default: 0.001)')
parser.add_argument('--triplet_weight', type=float, default=1.,
                    help='learning rate (default: 0.001)')
parser.add_argument('--drop_prob', type=float, default=.1,
                    help='learning rate (default: 0.001)')
parser.add_argument('--optim', type=str, default="adam",
                    help='learning rate (default: 0.001)')
parser.add_argument('--d_model', type=int, default=512,
                    help='learning rate (default: 0.001)')
parser.add_argument('--num_layer', type=int, default=6,
                    help='learning rate (default: 0.001)')
parser.add_argument('--downsample_num_upper', type=int, default=36,
                    help='learning rate (default: 0.001)')
parser.add_argument('--downsample_num_lower', type=int, default=28,
                    help='learning rate (default: 0.001)')
parser.add_argument('--noise_scale', type=float, default=0.,
                    help='learning rate (default: 0.001)')
parser.add_argument('--max_segments', type=int, default=15,
                    help='learning rate (default: 0.001)')
parser.add_argument("--runs_name", default="pixel_mask_encoder", type=str)
parser.add_argument("--project", default="mask autoencoder", type=str)

parser.add_argument(
    '--debug',
    action='store_true',
    help='use soft samples in training an inference',
)
parser.add_argument(
    '--transform',
    action='store_true',
    help='use soft samples in training an inference',
)
parser.add_argument('--num_bins', type=int, default=100,
                    help='learning rate (default: 0.001)')
parser.add_argument(
    '--shareloc',
    action='store_true',
    help='use soft samples in training an inference',
)
parser.add_argument('--mode', type=str, default="cls",
                    help='learning rate (default: 0.001)')
parser.add_argument(
    '--add_mapping',
    action='store_true',
    help='use soft samples in training an inference',
)
parser.add_argument(
    '--no_save',
    action='store_true',
    help='use soft samples in training an inference',
)
parser.add_argument(
    '--find_unused_parameters',
    action='store_true',
    help='use soft samples in training an inference',
)
parser.add_argument(
    '--attn_with_pos',
    action='store_true',
    help='use soft samples in training an inference',
)
parser.add_argument('--pos_embed_type', type=str, default="sine",
                    help='learning rate (default: 0.001)')
# Parse the command-line arguments
args = parser.parse_args()
if args.debug:
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
init_lr = args.lr
factor = args.factor
adam_eps = args.adam_eps
patience = args.patience
warmup = args.warmup
epoch = args.epoch
clip = args.clip
weight_decay = args.weight_decay
llm_embed = torch.load("/home/pirenjie/pretrained_weights/llava_embedding/llava_embedding.pt")


def visualize_polygons(gt_masks, generated_masks):
    fig, axs = plt.subplots(3, 6, figsize=(20, 10))

    for i, (gt_mask, generated_mask) in enumerate(zip(gt_masks , generated_masks)):
        if i >= 9:
            break
        ax = axs[i // 3, (i % 3) * 2]
        # ax = axs[i // 3, i % 3 + (i % 3) * 2-1 ]
        ax.imshow(gt_mask, cmap='gray')
        # Hide the top and right spines

        ax = axs[i // 3, (i % 3) * 2 + 1]
        # ax.set_ylim(ax.get_ylim()[::-1])
        # plot masks
        ax.imshow(generated_mask, cmap='gray')

        ax.legend()

    plt.tight_layout()
    plt.show()

def evaluate(model, refer, device):
    model.eval()
    generated_masks, gt_points = [], []
    ref_ids = refer.getRefIds(split='train')
    mask_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224), antialias=True)
    ])
    masks = []
    with torch.no_grad():
        for i, ref_id in enumerate(ref_ids):
            if len(generated_masks) > 9:
                break
            ref = refer.loadRefs(ref_id)[0]
            mask_pixel = refer.getMask_numpy(ref)#*255
            mask_pixel = mask_transform(mask_pixel).to(device).to(torch.float32).unsqueeze(dim=1)
            # calculate cls and reg losses
            generated = model(mask_pixel)
            # generated_masks.append((generated[0] >0.5).to(torch.float32).squeeze().cpu().numpy())
            generated_masks.append(generated[0].to(torch.float32).squeeze().sigmoid().cpu().numpy())
            masks.append(mask_pixel.squeeze().cpu().numpy())
        visualize_polygons(masks, generated_masks)




def run():
    device = torch.device('cuda')
    model = UNet(
        n_channels=1,
        n_classes=1
    )
    model = model.to(device)
    print(f'The model has {count_parameters(model):,} trainable parameters')
    model.load_state_dict(torch.load(
        f"/home/pirenjie/transformer-master/saved/unet_lr5e-4_noshortcut-model.pt"))
        # f"/home/pirenjie/transformer-master/saved/pixelmask_lr5e-4_200epoch_upper32_lower32_maxseg10_transform_mask01-model.pt"))
    refer = REFER(dataset='refcocog', splitBy='google', data_root="/home/pirenjie/data/refcoco")
    evaluate(model, refer, device)



if __name__ == '__main__':
    # Determine the number of GPUs available
    num_gpus = torch.cuda.device_count()
    print(f" world size {num_gpus}")
    run()
