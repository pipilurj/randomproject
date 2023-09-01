"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import math
import time
import torch
from torch import nn, optim
from torch.optim import Adam
import numpy as np
# from data import *
from models.model.transformer import Transformer, TransformerLoc
from util.bleu import idx_to_word, get_bleu
from util.epoch_timer import epoch_time
from boundingbox_dataset import BoundingBoxDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import argparse
import torch.nn.functional as F


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
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate (default: 0.001)')
parser.add_argument('--epoch', type=int, default=50,
                    help='learning rate (default: 0.001)')

# Parse the command-line arguments
args = parser.parse_args()
init_lr = args.lr
factor = 0.9
adam_eps = 5e-9
patience = 10
warmup = 100
epoch = args.epoch
clip = 1.0
weight_decay = 5e-4

model = TransformerLoc(
                    d_model=512,
                    max_len=128,
                    ffn_hidden=512,
                    n_head=8,
                    n_layers=6,
                    drop_prob=0.1,
                    device="cuda").cuda()

print(f'The model has {count_parameters(model):,} trainable parameters')
model.apply(initialize_weights)
optimizer = Adam(params=model.parameters(),
                 lr=init_lr,
                 weight_decay=weight_decay,
                 eps=adam_eps)

# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
#                                                  verbose=True,
#                                                  factor=factor,
#                                                  patience=patience)

criterion = nn.CrossEntropyLoss()
train_dataset = BoundingBoxDataset(100000)
val_dataset = BoundingBoxDataset(10000)
train_loader = DataLoader(train_dataset, batch_size=15000)
val_loader = DataLoader(val_dataset, batch_size=5000)

def train(model, train_loader, optimizer, criterion, clip):
    model.train()
    epoch_loss, epoch_cls_loss, epoch_reg_loss, epoch_triplet_loss = 0,0, 0, 0
    for i, (anchor, positive, negative, iou_positive, iou_negative) in enumerate(train_loader):
        optimizer.zero_grad()
        anchor, positive, negative = anchor.view(-1, 2, 2).cuda(), positive.view(-1, 2, 2).cuda(), negative.view(-1, 2, 2).cuda()
        # calculate cls and reg losses
        cls_pred, reg_pred = model(anchor.view(-1, 2, 2))
        cls_target = torch.zeros([len(cls_pred), 3]).cuda()
        cls_target[:, 0], cls_target[:, 1] = 0, 0
        cls_target[:, 2] = 1
        cls_pred_reshape = cls_pred.contiguous().view(-1, cls_pred.shape[-1])
        cls_target = cls_target.contiguous().view(-1).to(torch.long)

        loss_cls = criterion(cls_pred_reshape, cls_target)
        loss_reg = nn.L1Loss()(reg_pred[:, :2], anchor.view(-1, 2, 2))
        loss = loss_cls + loss_reg
        # calculate triplet loss
        anchor_embedding = model(anchor, return_embedding=True)
        positive_embedding = model(positive, return_embedding=True)
        negative_embedding = model(negative, return_embedding=True)
        distance_positive = torch.pairwise_distance(anchor_embedding, positive_embedding)
        distance_negative = torch.pairwise_distance(anchor_embedding, negative_embedding)
        triplet_loss = F.relu(distance_positive - distance_negative + 0.2).mean()

        loss += triplet_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
        epoch_cls_loss += loss_cls.item()
        epoch_reg_loss += loss_reg.item()
        epoch_triplet_loss += triplet_loss.item()

    print(f"+"*20)
    print(f"train epoch loss {epoch_loss / len(train_loader)}")
    print(f"train epoch cls loss {epoch_cls_loss / len(train_loader)}")
    print(f"train epoch reg loss {epoch_reg_loss / len(train_loader)}")
    print(f"train epoch triplet loss {epoch_triplet_loss / len(train_loader)}")
    for i, (pred, gt) in enumerate(zip(reg_pred[:, :2].view(-1, 4), anchor.view(-1, 4))):
        print(f"box pred {pred}")
        print(f"box gt {gt}")
        if i >1:
            break
    print(f"+"*20)
    return epoch_loss / len(train_loader), epoch_cls_loss / len(train_loader), epoch_reg_loss / len(train_loader), epoch_triplet_loss / len(train_loader)


def evaluate(model, val_loader, criterion):
    model.eval()
    epoch_loss, epoch_cls_loss, epoch_reg_loss, epoch_triplet_loss = 0,0, 0, 0
    with torch.no_grad():
        for i, (anchor, positive, negative, iou_positive, iou_negative) in enumerate(val_loader):
            anchor, positive, negative = anchor.view(-1, 2, 2).cuda(), positive.view(-1, 2, 2).cuda(), negative.view(-1, 2, 2).cuda()
            cls_pred, reg_pred = model(anchor)
            cls_target = torch.zeros([len(cls_pred), 3]).cuda()
            cls_target[:, 0], cls_target[:, 1] = 0, 0
            cls_target[:, 2] = 1
            cls_pred_reshape = cls_pred.contiguous().view(-1, cls_pred.shape[-1])
            cls_target = cls_target.contiguous().view(-1).to(torch.long)
            loss_cls = criterion(cls_pred_reshape, cls_target)
            loss_reg = nn.L1Loss()(reg_pred[:, :2], anchor)
            loss = loss_cls + loss_reg
            # calculate triplet loss
            anchor_embedding = model(anchor, return_embedding=True)
            positive_embedding = model(positive, return_embedding=True)
            negative_embedding = model(negative, return_embedding=True)
            distance_positive = torch.pairwise_distance(anchor_embedding, positive_embedding)
            distance_negative = torch.pairwise_distance(anchor_embedding, negative_embedding)
            triplet_loss = F.relu(distance_positive - distance_negative + 0.2).mean()

            loss += triplet_loss
            epoch_loss += loss.item()
            epoch_cls_loss += loss_cls.item()
            epoch_reg_loss += loss_reg.item()
            epoch_triplet_loss += triplet_loss.item()

    print(f"+"*20)
    print(f"val epoch loss {epoch_loss / len(val_loader)}")
    print(f"val epoch cls loss {epoch_cls_loss / len(val_loader)}")
    print(f"val epoch reg loss {epoch_reg_loss / len(val_loader)}")
    print(f"val epoch triplet loss {epoch_triplet_loss / len(val_loader)}")
    for i, (pred, gt) in enumerate(zip(reg_pred[:, :2].view(-1, 4), anchor.view(-1, 4))):
        print(f"box pred {pred}")
        print(f"box gt {gt}")
        if i >1:
            break
    print(f"+"*20)
    return epoch_loss / len(val_loader), epoch_cls_loss / len(val_loader), epoch_reg_loss / len(val_loader), epoch_triplet_loss / len(val_loader)


def run(total_epoch, best_loss):
    train_losses, train_losses_cls, train_losses_reg, train_losses_triplet, val_losses, val_losses_cls, val_losses_reg, val_losses_triplet = [], [], [], [], [], [], [], []
    for step in range(total_epoch):
        start_time = time.time()
        train_loss, train_loss_cls, train_loss_reg, train_loss_triplet = train(model, train_loader, optimizer, criterion, clip)
        # valid_loss, bleu = evaluate(model, valid_iter, criterion)
        end_time = time.time()

        # if step > warmup:
        #     scheduler.step(valid_loss)

        train_losses.append(train_loss)
        train_losses_cls.append(train_loss_cls)
        train_losses_reg.append(train_loss_reg)
        train_losses_triplet.append(train_loss_triplet)

        val_loss, val_loss_cls, val_loss_reg, val_loss_triplet = evaluate(model, val_loader, criterion)
        val_losses.append(val_loss)
        val_losses_cls.append(val_loss_cls)
        val_losses_reg.append(val_loss_reg)
        val_losses_triplet.append(val_loss_triplet)

        # test_losses.append(valid_loss)
        # bleus.append(bleu)

        # if valid_loss < best_loss:
        #     best_loss = valid_loss
        #     torch.save(model.state_dict(), 'saved/model-{0}.pt'.format(valid_loss))

        # f = open('result/train_loss.txt', 'w')
        # f.write(str(train_losses))
        # f.close()
        #
        # f = open('result/bleu.txt', 'w')
        # f.write(str(bleus))
        # f.close()
        #
        # f = open('result/test_loss.txt', 'w')
        # f.write(str(test_losses))
        # f.close()
        #
        # print(f'Epoch: {step + 1} | Time: {epoch_mins}m {epoch_secs}s')
        # print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        # print(f'\tVal Loss: {valid_loss:.3f} |  Val PPL: {math.exp(valid_loss):7.3f}')
        # print(f'\tBLEU Score: {bleu:.3f}')
    # Sample training and validation loss values

    # Create a figure and subplots
        fig, axs = plt.subplots(2, 3, figsize=(10, 6))
        training_losses = [train_losses, train_losses_cls, train_losses_reg, train_losses_triplet]
        validation_losses = [val_losses, val_losses_cls, val_losses_reg, val_losses_triplet]
        keys = ["avg", "cls", "reg", "triplet"]
        # Plot training losses
        for i in range(4):
            axs[0, i].plot(np.arange(len(training_losses[i])), training_losses[i])
            axs[0, i].set_title(f"Training {keys[i]} Loss {i+1}")
            axs[0, i].set_xlabel('Epoch')
            axs[0, i].set_ylabel('Loss')

        # Plot validation losses
        for i in range(4):
            axs[1, i].plot(np.arange(len(validation_losses[i])), validation_losses[i])
            axs[1, i].set_title(f"Validation {keys[i]} Loss {i+1}")
            axs[1, i].set_xlabel('Epoch')
            axs[1, i].set_ylabel('Loss')

        # Adjust spacing between subplots
        plt.tight_layout()

        # Save the figure as an image
        plt.savefig(args.figname)

        # Show the plot
        plt.show()

if __name__ == '__main__':
    run(total_epoch=epoch, best_loss=10000)
