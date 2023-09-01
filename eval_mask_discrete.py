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
from models.model.transformer_discrete import TransformerDiscrete
from segmentation_discrete_dataset import SegmentationDiscreteDataset, custom_collate_fn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import argparse
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from focal_loss import focal_loss
from torch.optim.lr_scheduler import CosineAnnealingLR


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
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
init_lr = args.lr
factor = args.factor
adam_eps = args.adam_eps
patience = args.patience
warmup = args.warmup
epoch = args.epoch
clip = args.clip
weight_decay = args.weight_decay
llm_embed = torch.load("/home/pirenjie/pretrained_weights/llava_embedding/llava_embedding.pt")


def visualize_polygons(gt_polygons, pred_polygons):
    fig, axs = plt.subplots(3, 3, figsize=(10, 10))

    for i, (gt_polygon, pred_polygon) in enumerate(zip(gt_polygons, pred_polygons)):
        ax = axs[i // 3, i % 3]
        ax.plot(gt_polygon[:, 0], gt_polygon[:, 1], color='red', label='Ground Truth')
        ax.plot(pred_polygon[:, 0], pred_polygon[:, 1], color='blue', label='Prediction')

        # Set x and y axis limits to [0, 1]
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        # Connect the first and last point of each polygon
        ax.plot([gt_polygon[0, 0], gt_polygon[-1, 0]], [gt_polygon[0, 1], gt_polygon[-1, 1]], color='red')
        ax.plot([pred_polygon[0, 0], pred_polygon[-1, 0]], [pred_polygon[0, 1], pred_polygon[-1, 1]], color='blue')

        ax.legend()

    plt.tight_layout()
    plt.show()

def evaluate(model, val_loader, criterion, device):
    model.eval()
    epoch_loss, epoch_cls_loss, epoch_reg_loss, epoch_triplet_loss = 0, 0, 0, 0
    generated_points, gt_points = [], []
    with torch.no_grad():
        for i, (anchor_input_enc, anchor_input_dec, anchor_targets, anchor_masks_enc, anchor_masks_dec, positive_input_enc,
                positive_input_dec,
                positive_targets, positive_masks_enc, positive_masks_dec, negative_input_enc, negative_input_dec,
                negative_targets,
                negative_masks_enc, negative_masks_dec) in enumerate(val_loader):
            anchor_input_enc, anchor_input_dec, anchor_targets, anchor_masks_enc, anchor_masks_dec, positive_input_enc, positive_input_dec, positive_targets, positive_masks_enc, positive_masks_dec, negative_input_enc, negative_input_dec, negative_targets, \
            negative_masks_enc, negative_masks_dec = \
                anchor_input_enc.to(device), anchor_input_dec.to(device), anchor_targets.to(device), anchor_masks_enc.to(
                    device), anchor_masks_dec.to(device), positive_input_enc.to(device), positive_input_dec.to(
                    device), positive_targets.to(device), positive_masks_enc.to(device), positive_masks_dec.to(
                    device), negative_input_enc.to(device), negative_input_dec.to(device), negative_targets.to(
                    device), negative_masks_enc.to(device), negative_masks_dec.to(device)
            # calculate cls and reg losses
            outputs = model(anchor_input_enc, anchor_input_dec, anchor_masks_enc, anchor_masks_dec)
            cls_pred = outputs["output"]
            hidden_repr = outputs["hidden_repr"]
            cls_pred_reshape = cls_pred.contiguous().view(-1, cls_pred.shape[-1])
            cls_target = anchor_targets.contiguous().view(-1).to(torch.long)

            loss = criterion(cls_pred_reshape[cls_target != -1], cls_target[cls_target != -1]) * args.cls_weight
            epoch_loss += loss.item()
            epoch_cls_loss += torch.tensor(0.).item()
            epoch_reg_loss += torch.tensor(0.).item()
            epoch_triplet_loss += torch.tensor(0.).item()
            generated = model.generate(anchor_input_enc, masks_enc=anchor_masks_enc)
            generated_points.append(model.convert_to_continuous(generated[0].cpu().numpy() , 100))
            gt_points.append(model.convert_to_continuous(anchor_targets[0][anchor_targets[0] != -1][:-1].cpu().numpy() , 100))
            if len(gt_points) >= 9:
                break
        visualize_polygons(gt_points, generated_points)

    print(f"+" * 20)
    print(f"val epoch loss {epoch_loss / len(val_loader)}")
    print(f"val epoch cls loss {epoch_cls_loss / len(val_loader)}")
    print(f"val epoch reg loss {epoch_reg_loss / len(val_loader)}")
    print(f"val epoch triplet loss {epoch_triplet_loss / len(val_loader)}")
    print(f"box pred {cls_pred[0].argmax(-1)[anchor_targets[0] != -1].flatten()}")
    print(f"box gt {anchor_targets[0][anchor_targets[0] != -1].flatten()}")
    print(f"+" * 20)
    return epoch_loss / len(val_loader), epoch_cls_loss / len(val_loader), epoch_reg_loss / len(
        val_loader), epoch_triplet_loss / len(val_loader)




def run():
    device = torch.device('cuda')
    model = TransformerDiscrete(
        d_model=args.d_model,
        max_len=70,
        ffn_hidden=args.d_model,
        n_head=8,
        n_layers=args.num_layer,
        drop_prob=args.drop_prob,
        mode=args.mode,
        share_loc_embed=args.shareloc,
        num_bins=args.num_bins,
        device="cuda")
    model = model.to(device)
    print(f'The model has {count_parameters(model):,} trainable parameters')
    if args.pretrained is None:
        model.apply(initialize_weights)
    else:
        model.load_state_dict(torch.load(args.pretrained))
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    train_dataset = SegmentationDiscreteDataset(path="/home/pirenjie/data/coco/annotations/instances_train2017.json",
                                                transformation=args.transform, size=args.train_size,
                                                num_bins=args.num_bins, downsample_num_lower=args.downsample_num_lower, downsample_num_upper=args.downsample_num_upper)
    train_loader = DataLoader(train_dataset, batch_size=1,
                              collate_fn=custom_collate_fn, num_workers=4, pin_memory=True)
    val_dataset = SegmentationDiscreteDataset(path="/home/pirenjie/data/coco/annotations/instances_val2017.json",
                                              transformation=args.transform, size=args.val_size, num_bins=args.num_bins, downsample_num_lower=args.downsample_num_lower, downsample_num_upper=args.downsample_num_upper)
    val_loader = DataLoader(val_dataset, batch_size=args.val_bs, collate_fn=custom_collate_fn, num_workers=4)
    val_loss, val_loss_cls, val_loss_reg, val_loss_triplet = evaluate(model, train_loader, criterion, device)



if __name__ == '__main__':
    # Determine the number of GPUs available
    num_gpus = torch.cuda.device_count()
    print(f" world size {num_gpus}")
    run()
