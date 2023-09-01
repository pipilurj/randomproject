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
from models.model.vae import VAE
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
import wandb
from dice_loss import dice_loss
os.system("wandb login --relogin 8bb1fef7b4815daa3cb2ec7c5b0b9ee40d7ea6ed")


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
parser.add_argument('--clip', type=float, default=3.0,
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
parser.add_argument('--recon_loss', type=str, default="cross_entropy",
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
parser.add_argument('--noise_scale', type=float, default=0.,
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
parser.add_argument('--feature_dim', type=int, default=4096,
                    help='learning rate (default: 0.001)')
parser.add_argument('--latent_dim', type=int, default=4096,
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
parser.add_argument('--transform_prob', type=float, default=0.5,
                    help='learning rate (default: 0.001)')
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


def kl_divergence(mean_x, var_x, mean_y, var_y):
    loss = 0.5 * (torch.log(var_y / var_x) + (var_x + (mean_x - mean_y) ** 2) / var_y - 1)
    return loss.mean()


def train(model, train_loader, optimizer, criterion, clip, device, rank, n_gpus):
    llm_embed_mean, llm_embed_var = llm_embed.mean().unsqueeze(dim=0).to(device), llm_embed.var().unsqueeze(dim=0).to(
        device)
    model.train()
    total_loss, total_rec_loss, total_dist_loss = 0.0, 0.0, 0.0
    total_samples = 0
    cur_iters = 0
    for i, mask_pixel in enumerate(train_loader):
        cur_iters += 1
        mask_pixel = mask_pixel.to(device)
        # calculate cls and reg losses
        reconstruction, mu, logvar = model(mask_pixel)
        if args.recon_loss == "cross_entropy":
            dice = dice_loss(F.sigmoid(reconstruction.squeeze(1)), mask_pixel.squeeze(1).float(), multiclass=False)
            reconstruction = reconstruction.view(-1)
            target = mask_pixel.view(-1)
            loss_reconstruction = F.binary_cross_entropy_with_logits(reconstruction, target) + dice
        else:
            loss_reconstruction = criterion(reconstruction, mask_pixel) + dice
        # loss_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) * args.dist_weight
        loss = loss_reconstruction
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total_loss += loss.item()
        total_rec_loss += loss_reconstruction.item()
        total_dist_loss += 0
        total_samples += 1
        if (i % 20 == 0 or i == len(train_loader) - 1) and rank == 0:
            print(f"+" * 20)
            print(f"train epoch loss {total_loss / total_samples}")
            print(f"train epoch rec loss {total_rec_loss / total_samples}")
            print(f"train epoch dist loss {total_dist_loss / total_samples}")
            print(f"+" * 20)
    if n_gpus > 1:
        dist.reduce(torch.tensor(total_loss, device=device), dst=0)
        dist.reduce(torch.tensor(total_dist_loss, device=device), dst=0)
        dist.reduce(torch.tensor(total_rec_loss, device=device), dst=0)
        dist.reduce(torch.tensor(total_samples, device=device), dst=0)
        if dist.get_rank() == 0:
            average_loss = total_loss / total_samples
            average_rec_loss = total_rec_loss / total_samples
            average_dist_loss = total_dist_loss / total_samples
            return average_loss, average_rec_loss, average_dist_loss
        else:
            return None, None, None
    else:
        return total_loss / total_samples, total_rec_loss / total_samples, total_dist_loss / total_samples


def evaluate(model, val_loader, criterion, device, n_gpus):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for i, mask_pixel in enumerate(val_loader):
            mask_pixel = mask_pixel.to(device)
            # calculate cls and reg losses
            reconstruction, mu, logvar = model(mask_pixel)
            if args.recon_loss == "cross_entropy":
                reconstruction = reconstruction.view(-1)
                target = mask_pixel.view(-1)
                loss_reconstruction = (-target * torch.log(reconstruction + 1e-10) - (1 - target) * torch.log(1 - reconstruction + 1e-10)).mean()
            else:
                loss_reconstruction = criterion(reconstruction, mask_pixel)
            loss = loss_reconstruction
            total_loss += loss.item()
            total_samples += 1
    if n_gpus > 1:
        dist.reduce(torch.tensor(total_loss, device=device), dst=0)
        dist.reduce(torch.tensor(total_samples, device=device), dst=0)
        if dist.get_rank() == 0:
            average_loss = total_loss / total_samples
            print(f"val epoch loss {average_loss}")
            return average_loss
        else:
            return None
    else:
        return total_loss / total_samples


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
    if local_rank == 0:
        run = wandb.init(project=args.project, name=args.runs_name)
    model = VAE(
        feature_dim=args.feature_dim,
        latent_dim=args.latent_dim,
        noise_scale=args.noise_scale
    )
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
    criterion = nn.MSELoss()
    criterion = nn.BCEWithLogitsLoss()
    train_dataset = SegmentationMaskDataset(path="/home/pirenjie/data/coco/annotations/instances_train2017.json",
                                            mask_path="/home/pirenjie/data/coco/masks/train",
                                            transformation=args.transform, transform_prob=args.transform_prob,
                                            size=args.train_size,
                                            num_bins=args.num_bins, downsample_num_lower=args.downsample_num_lower,
                                            downsample_num_upper=args.downsample_num_upper,
                                            max_segments=args.max_segments)
    val_dataset = SegmentationMaskDataset(path="/home/pirenjie/data/coco/annotations/instances_val2017.json",
                                          mask_path="/home/pirenjie/data/coco/masks/val",
                                          transformation=False, size=args.val_size, num_bins=args.num_bins,
                                          downsample_num_lower=args.downsample_num_lower,
                                          downsample_num_upper=args.downsample_num_upper,
                                          max_segments=args.max_segments)
    if n_gpus > 1:
        sampler_train = torch.utils.data.distributed.DistributedSampler(train_dataset)
        sampler_val = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        sampler_train = None
        sampler_val = None
    train_loader = DataLoader(train_dataset, batch_size=args.train_bs // n_gpus, sampler=sampler_train, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.val_bs // n_gpus, sampler=sampler_val, num_workers=4)
    train_losses, train_losses_cls, train_losses_dist, val_losses = [], [], [], []
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epoch)
    for step in range(args.epoch):
        start_time = time.time()
        train_loss, train_cls_loss, train_dist_loss = train(model, train_loader, optimizer, criterion, clip, device,
                                                            local_rank, n_gpus)
        scheduler.step()
        train_losses.append(train_loss)
        train_losses_cls.append(train_cls_loss)
        train_losses_dist.append(train_dist_loss)
        # if local_rank == 0:
        val_loss = evaluate(model, val_loader, criterion, device, n_gpus)
        val_losses.append(val_loss)
        if local_rank == 0:
            if not args.debug:
                wandb.log({"train_loss": train_loss,
                           "train_cls_loss": train_cls_loss,
                           "train_dist_loss": train_dist_loss,
                           "val_loss": val_loss
                           }, step=step)
            if not args.no_save:
                if n_gpus > 1:
                    torch.save(model.module.state_dict(), f'saved/{args.runs_name}-model.pt')
                else:
                    torch.save(model.state_dict(), f'saved/{args.runs_name}-model.pt')
            fig, axs = plt.subplots(2, 1, figsize=(12, 6))
            keys = ["loss"]
            # Plot training losses
            # for i in range(1):
            axs[0].plot(np.arange(len(train_losses)), train_losses)
            axs[0].set_title(f"Training Loss")
            axs[0].set_xlabel('Epoch')
            axs[0].set_ylabel('Loss')

            # Plot validation losses
            # for i in range(1):
            axs[1].plot(np.arange(len(val_losses)), val_losses)
            axs[1].set_title(f"Validation Loss")
            axs[1].set_xlabel('Epoch')
            axs[1].set_ylabel('Loss')

            # Adjust spacing between subplots
            plt.tight_layout()

            # Save the figure as an image
            plt.savefig(args.figname)
            plt.close()
            # Show the plot
            # plt.show()
    if local_rank == 0:
        run.finish()


if __name__ == '__main__':
    # Determine the number of GPUs available
    num_gpus = torch.cuda.device_count()
    print(f" world size {num_gpus}")
    run()
