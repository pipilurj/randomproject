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
from segmentation_mask_dataset import getIoU, SegmentationMaskDataset
import pycocotools.mask as maskUtils

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
parser.add_argument('--max_segments', type=int, default=10,
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


def visualize_polygons(masks, pred_polygons):
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

    plt.tight_layout()
    plt.show()

def visualize_iou_loss(ious, losses, noise_rates):
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))

    ax = axs[0]

    # Hide the top and right spines
    ax.semilogx(noise_rates, ious, color='blue', label='iou')

    ax = axs[1]
    # ax.set_ylim(ax.get_ylim()[::-1])
    # plot masks
    ax.semilogx(noise_rates, losses, color='blue', label='iou')
    plt.tight_layout()
    for ax in axs:
        ax.set_xticks(noise_rates)
        ax.set_xticklabels(noise_rates)

    # Add grid to both subplots
    for ax in axs:
        ax.grid(True)
    plt.savefig(args.figname)
    plt.show()

def evaluate_ref(model, refer, device):
    model.eval()
    generated_points, gt_points = [], []
    ref_ids = refer.getRefIds(split='train')
    mask_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224), antialias=True)
    ])
    masks = []
    with torch.no_grad():
        for i,  ref_id in enumerate(ref_ids):
            if len(generated_points) > 9:
                break
            ref = refer.loadRefs(ref_id)[0]
            mask_pixel = refer.getMask_numpy(ref)#*255
            mask_pixel = mask_transform(mask_pixel).to(device).to(torch.float32).unsqueeze(dim=1)
            # calculate cls and reg losses
            generated = model.generate(mask_pixel)

            generated_points.append(generated[0])
            masks.append(mask_pixel.squeeze().cpu().numpy())
        visualize_polygons(masks, generated_points)


def evaluate_coco(model, val_loader, criterion, device, n_gpus, noise_rate=0.):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    pred_masks, gt_masks = [], []
    ious = []
    with torch.no_grad():
        for i, (inputs_dec, mask_pixel, targets, masks_dec) in enumerate(val_loader):
            inputs_dec, mask_pixel, targets, masks_dec = \
                inputs_dec.to(device), mask_pixel.to(device), targets.to(device), masks_dec.to(device)
            # calculate cls and reg losses
            outputs = model(input_dec=inputs_dec, mask_pixel=mask_pixel, mask_dec=masks_dec)
            cls_pred = outputs["output"]
            hidden_repr = outputs["hidden_repr"]
            cls_pred_reshape = cls_pred.contiguous().view(-1, cls_pred.shape[-1])
            cls_target = targets.contiguous().view(-1).to(torch.long)
            loss = criterion(cls_pred_reshape[cls_target != -1], cls_target[cls_target != -1]) * args.cls_weight
            total_loss += loss.item()
            total_samples += 1
            enc_feature = model(mask_pixel=mask_pixel, return_embedding=True)
            enc_feature = enc_feature + noise_rate * torch.randn_like(enc_feature)
            generated_sequence = model.generate(encoder_repr=enc_feature)
            for mask_gt, sequence in zip(mask_pixel, generated_sequence):
                sequence = [s.flatten()*224 for s in sequence]
                if all([len(s) > 4 for s in sequence]):
                    pred_rles = maskUtils.frPyObjects(
                        sequence, 224, 224)  # list[rle]
                    pred_rle = maskUtils.merge(pred_rles)
                    pred_mask = maskUtils.decode(pred_rle)[None]
                    pred_mask = np.array(pred_mask).astype(np.float32)
                    pred_masks.append(pred_mask)
                    gt_masks.append(mask_gt.cpu().numpy())
                    # iou = getIoU(mask_gt.cpu().numpy(), pred_mask)
                else:
                    pred_masks.append(torch.zeros_like(mask_gt).cpu().numpy())
                    gt_masks.append(mask_gt.cpu().numpy())
                # ious.append(iou)
            # pred_masks.append(pred_rle)
    ious = getIoU(np.stack(pred_masks),np.stack(gt_masks))
    ious_len = len(pred_masks)
    ious_sum = ious*ious_len
    if n_gpus > 1:
        dist.reduce(torch.tensor(total_loss, device=device), dst=0)
        dist.reduce(torch.tensor(total_samples, device=device), dst=0)
        dist.reduce(torch.tensor(ious_sum, device=device), dst=0)
        dist.reduce(torch.tensor(ious_len, device=device), dst=0)
        if dist.get_rank() == 0:
            average_loss = total_loss / total_samples
            average_iou = ious_sum / ious_len
            print(f"val epoch loss {average_loss}")
            print(f"box pred {cls_pred[0].argmax(-1)[targets[0] != -1].flatten()}")
            print(f"box gt {targets[0][targets[0] != -1].flatten()}")
            return average_loss, average_iou
        else:
            return None, None
    else:
        return total_loss / total_samples, ious_sum/ious_len

def setup_distributed():
    torch.cuda.set_device(torch.cuda.current_device())
    dist.init_process_group(backend='nccl', init_method='env://')
def run():
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        if n_gpus > 1:
            setup_distributed()
            local_rank = torch.distributed.get_rank()
            torch.cuda.set_device(local_rank)
            device = torch.device('cuda', local_rank)
            torch.cuda.manual_seed(0)
        else:
            local_rank = 0
            device = torch.device('cuda')
    else:
        local_rank = 0
        device = torch.device('cpu')
    model = TransformerMask(
        d_model=512,
        max_len=512,
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
    # refer = REFER(dataset='refcocog', splitBy='google', data_root="/home/pirenjie/data/refcoco")
    # evaluate(model, refer, device)
    # train_dataset = SegmentationMaskDataset(path="/home/pirenjie/data/coco/annotations/instances_train2017.json",
    #                                         mask_path="/home/pirenjie/data/coco/masks/train",
    #                                         transformation=args.transform, transform_prob=args.transform_prob,
    #                                         size=args.train_size,
    #                                         num_bins=args.num_bins, downsample_num_lower=args.downsample_num_lower,
    #                                         downsample_num_upper=args.downsample_num_upper,
    #                                         max_segments=args.max_segments)
    val_dataset = SegmentationMaskDataset(path="/home/pirenjie/data/coco/annotations/instances_val2017.json",
                                          mask_path="/home/pirenjie/data/coco/masks/val",
                                          transformation=False, size=args.val_size, num_bins=args.num_bins,
                                          downsample_num_lower=args.downsample_num_lower,
                                          downsample_num_upper=args.downsample_num_upper,
                                          max_segments=args.max_segments)
    if n_gpus > 1:
        # sampler_train = torch.utils.data.distributed.DistributedSampler(train_dataset)
        sampler_val = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        sampler_train = None
        sampler_val = None
    # train_loader = DataLoader(train_dataset, batch_size=args.train_bs // n_gpus, sampler=sampler_train,
    #                           collate_fn=custom_collate_fn, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.val_bs // n_gpus, sampler=sampler_val,
                            collate_fn=custom_collate_fn, num_workers=4)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    noise_rates = [0., 1e-7 , 1e-6 , 1e-5, 1e-4, 1e-3, 1e-2, 2.5e-2, 5e-2, 1e-1, 1]
    # noise_rates = [1e-2, 5e-2, 1e-1, 1]
    losses, ious = [], []
    for noise_rate in noise_rates:
        val_loss, iou = evaluate_coco(model, val_loader, criterion, device, n_gpus, noise_rate=noise_rate)
        losses.append(val_loss)
        ious.append(iou)
    print(losses)
    print(ious)
    if local_rank == 0:
        visualize_iou_loss(ious, losses, noise_rates)


if __name__ == '__main__':
    # Determine the number of GPUs available
    num_gpus = torch.cuda.device_count()
    print(f" world size {num_gpus}")
    run()
