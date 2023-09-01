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
parser.add_argument('--val_bs', type=int, default=10000,
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


def train(model, train_loader, optimizer, criterion, clip, device, rank):
    llm_embed_mean, llm_embed_var = llm_embed.mean(dim=0).to(device), llm_embed.var(dim=0).to(device)
    model.train()
    epoch_loss, epoch_cls_loss, epoch_reg_loss, epoch_triplet_loss, epoch_dist_loss = 0, 0, 0, 0, 0
    # for i, (anchor, positive, negative, iou_positive, iou_negative) in enumerate(train_loader):
    cur_iters = 0
    for i, (anchor_input_enc, anchor_input_dec, anchor_targets, anchor_masks_enc, anchor_masks_dec, positive_input_enc,
            positive_input_dec,
            positive_targets, positive_masks_enc, positive_masks_dec, negative_input_enc, negative_input_dec,
            negative_targets,
            negative_masks_enc, negative_masks_dec) in enumerate(train_loader):
        cur_iters += 1
        optimizer.zero_grad()
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
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
        epoch_cls_loss += torch.tensor(0.).item()
        epoch_reg_loss += torch.tensor(0.).item()
        epoch_triplet_loss += torch.tensor(0.).item()
        epoch_dist_loss += torch.tensor(0.).item()
        if (i % 20 == 0 or i == len(train_loader) - 1) and rank == 0:
            print(f"+" * 20)
            print(f"train epoch loss {epoch_loss / cur_iters}")
            print(f"train epoch cls loss {epoch_cls_loss / cur_iters}")
            print(f"train epoch reg loss {epoch_reg_loss / cur_iters}")
            print(f"train epoch triplet loss {epoch_triplet_loss / cur_iters}")
            print(f"train epoch dist loss {epoch_dist_loss / cur_iters}")
            print(f"box pred {cls_pred[0].argmax(-1)[anchor_targets[0] != -1].flatten()}")
            print(f"box gt {anchor_targets[0][anchor_targets[0] != -1].flatten()}")
            print(f"+" * 20)
    return epoch_loss / len(train_loader), epoch_cls_loss / len(train_loader), epoch_reg_loss / len(
        train_loader), epoch_triplet_loss / len(train_loader), epoch_dist_loss / len(train_loader)


def evaluate(model, val_loader, criterion, device):
    model.eval()
    epoch_loss, epoch_cls_loss, epoch_reg_loss, epoch_triplet_loss = 0, 0, 0, 0
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
        device = torch.device('cpu')
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
    if n_gpus > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=args.find_unused_parameters)
    if args.optim == "adam":
        optimizer = Adam(params=model.parameters(),
                         lr=init_lr,
                         weight_decay=weight_decay,
                         eps=adam_eps)
    else:
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08,
                                      weight_decay=5e-4)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
    #                                                  verbose=True,
    #                                                  factor=factor,
    #                                                  patience=patience)
    if args.cls_loss == "cross_entropy":
        criterion = nn.CrossEntropyLoss(ignore_index=-1)
    else:
        criterion = focal_loss()
    train_dataset = SegmentationDiscreteDataset(path="/home/pirenjie/data/coco/annotations/instances_train2017.json",
                                                transformation=args.transform, size=args.train_size,
                                                num_bins=args.num_bins, downsample_num_lower=args.downsample_num_lower, downsample_num_upper=args.downsample_num_upper)
    if n_gpus > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        sampler = None
    train_loader = DataLoader(train_dataset, batch_size=args.train_bs // n_gpus, sampler=sampler,
                              collate_fn=custom_collate_fn, num_workers=4, pin_memory=True)
    val_dataset = SegmentationDiscreteDataset(path="/home/pirenjie/data/coco/annotations/instances_val2017.json",
                                              transformation=args.transform, size=args.val_size, num_bins=args.num_bins, downsample_num_lower=args.downsample_num_lower, downsample_num_upper=args.downsample_num_upper)
    val_loader = DataLoader(val_dataset, batch_size=args.val_bs, collate_fn=custom_collate_fn, num_workers=4)
    train_losses, train_losses_cls, train_losses_reg, train_losses_triplet, train_losses_dist, val_losses, val_losses_cls, val_losses_reg, val_losses_triplet = [], [], [], [], [], [], [], [], []
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epoch)
    for step in range(args.epoch):
        start_time = time.time()
        train_loss, train_loss_cls, train_loss_reg, train_loss_triplet, train_loss_dist = train(model, train_loader,
                                                                                                optimizer,
                                                                                                criterion, clip, device,
                                                                                                local_rank)
        scheduler.step()
        train_losses.append(train_loss)
        train_losses_cls.append(train_loss_cls)
        train_losses_reg.append(train_loss_reg)
        train_losses_triplet.append(train_loss_triplet)
        train_losses_dist.append(train_loss_dist)

        if local_rank == 0:
            val_loss, val_loss_cls, val_loss_reg, val_loss_triplet = evaluate(model, val_loader, criterion, device)
            val_losses.append(val_loss)
            val_losses_cls.append(val_loss_cls)
            val_losses_reg.append(val_loss_reg)
            val_losses_triplet.append(val_loss_triplet)
        if local_rank == 0:
            if not args.no_save:
                if n_gpus > 1:
                    torch.save(model.module.state_dict(), f'saved/{args.figname}-model.pt')
                else:
                    torch.save(model.state_dict(), f'saved/{args.figname}-model.pt')
            fig, axs = plt.subplots(2, 5, figsize=(12, 6))
            training_losses = [train_losses, train_losses_cls, train_losses_reg, train_losses_triplet,
                               train_losses_dist]
            validation_losses = [val_losses, val_losses_cls, val_losses_reg, val_losses_triplet]
            keys = ["avg", "cls", "reg", "triplet", "dist"]
            # Plot training losses
            for i in range(5):
                axs[0, i].plot(np.arange(len(training_losses[i])), training_losses[i])
                axs[0, i].set_title(f"Training {keys[i]} Loss {i + 1}")
                axs[0, i].set_xlabel('Epoch')
                axs[0, i].set_ylabel('Loss')

            # Plot validation losses
            for i in range(4):
                axs[1, i].plot(np.arange(len(validation_losses[i])), validation_losses[i])
                axs[1, i].set_title(f"Validation {keys[i]} Loss {i + 1}")
                axs[1, i].set_xlabel('Epoch')
                axs[1, i].set_ylabel('Loss')

            # Adjust spacing between subplots
            plt.tight_layout()

            # Save the figure as an image
            plt.savefig(args.figname)
            plt.close()
            # Show the plot
            # plt.show()


if __name__ == '__main__':
    # Determine the number of GPUs available
    num_gpus = torch.cuda.device_count()
    print(f" world size {num_gpus}")
    run()
