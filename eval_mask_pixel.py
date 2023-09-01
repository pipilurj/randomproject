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
from models.model.transformer_mask import TransformerMask
from segmentation_mask_dataset import SegmentationMaskDataset, custom_collate_fn
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


def visualize_polygons(masks, pred_polygons, noise_rate, l1_loss, l2_loss):
    fig, axs = plt.subplots(3, 6, figsize=(20, 10))

    for i, (mask, pred_polygon) in enumerate(zip(masks , pred_polygons)):
        if i >= 9:
            break
        ax = axs[i // 3, (i % 3) * 2]
        # ax = axs[i // 3, i % 3 + (i % 3) * 2-1 ]
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_ylim(ax.get_ylim()[::-1])

        # Hide the top and right spines
        for poly in pred_polygon:
            # ax.plot(poly[:, 0], poly[:, 1], "o", color='blue', label='Prediction')
            # ax.plot([poly[0, 0], poly[-1, 0]], [poly[0, 1], poly[-1, 1]], "o", color='blue')
            ax.plot(poly[:, 0], poly[:, 1], color='blue', label='Prediction')
            ax.plot([poly[0, 0], poly[-1, 0]], [poly[0, 1], poly[-1, 1]], color='blue')

        ax = axs[i // 3, (i % 3) * 2 + 1]
        # ax.set_ylim(ax.get_ylim()[::-1])
        # plot masks
        ax.imshow(mask, cmap='gray')

        ax.legend()
    fig.suptitle(f"noise rate {noise_rate}, l1 loss {l1_loss}, l2 loss {l2_loss}", fontsize=32)
    plt.tight_layout()
    plt.show()

def evaluate(model, refer, device, noise_rate=0.):
    model.eval()
    generated_points, gt_points = [], []
    ref_ids = refer.getRefIds(split='train')
    mask_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224), antialias=True)
    ])
    masks = []
    with torch.no_grad():
        l1_loss, l2_loss = 0, 0
        for i,  ref_id in enumerate(ref_ids):
            if len(generated_points) > 9:
                break
            ref = refer.loadRefs(ref_id)[0]
            mask_pixel = refer.getMask_numpy(ref)#*255
            mask_pixel = mask_transform(mask_pixel).to(device).to(torch.float32).unsqueeze(dim=1)
            # calculate cls and reg losses
            enc_feature = model(mask_pixel=mask_pixel, return_embedding=True)
            enc_feature_noise = enc_feature + noise_rate * torch.randn_like(enc_feature)
            l1_loss += F.l1_loss(enc_feature, enc_feature_noise)
            l2_loss += F.mse_loss(enc_feature, enc_feature_noise)
            # generated_sequence = model.generate(encoder_repr=enc_feature)
            generated_sequence = model.generate(encoder_repr=enc_feature_noise)

            generated_points.append(generated_sequence[0])
            masks.append(mask_pixel.squeeze().cpu().numpy())
        print(f"noise rate {noise_rate}, l1 loss {round((l1_loss/len(generated_points)).item(), 5)}, l2 loss {round((l2_loss/len(generated_points)).item(), 5)}")
        visualize_polygons(masks, generated_points, noise_rate, l1_loss/len(generated_points), l2_loss/len(generated_points))




def run():
    device = torch.device('cuda')
    model = TransformerMask(
        d_model=512,
        max_len=500,
        ffn_hidden=512,
        n_head=8,
        n_layers=6,
        add_mapping=False,
        # add_mapping=True,
        num_bins=100,
        mode="cls",
        drop_prob=0.1,
        share_loc_embed=True,
        device="cuda")
    model = model.to(device)
    print(f'The model has {count_parameters(model):,} trainable parameters')
    model.load_state_dict(torch.load(
        f"/home/pirenjie/transformer-master/saved/distributed_run_mask_pixel.jpg-model.pt"))
        # f"/home/pirenjie/transformer-master/saved/pixelmask_lr5e-4_200epoch_upper32_lower32_maxseg10_transform_mask01-model.pt"))
    refer = REFER(dataset='refcocog', splitBy='google', data_root="/home/pirenjie/data/refcoco")
    evaluate(model, refer, device, noise_rate=0.0)
    evaluate(model, refer, device, noise_rate=0.001)
    evaluate(model, refer, device, noise_rate=0.005)
    evaluate(model, refer, device, noise_rate=0.01)
    evaluate(model, refer, device, noise_rate=0.02)
    evaluate(model, refer, device, noise_rate=0.03)
    evaluate(model, refer, device, noise_rate=0.04)
    evaluate(model, refer, device, noise_rate=0.05)
    evaluate(model, refer, device, noise_rate=0.06)
    evaluate(model, refer, device, noise_rate=0.07)
    evaluate(model, refer, device, noise_rate=0.08)
    evaluate(model, refer, device, noise_rate=0.09)
    evaluate(model, refer, device, noise_rate=0.1)



if __name__ == '__main__':
    # Determine the number of GPUs available
    num_gpus = torch.cuda.device_count()
    print(f" world size {num_gpus}")
    run()
