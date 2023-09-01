"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn
from models.embedding.token_embeddings import TokenEmbedding
from segmentation_mask_dataset import SegmentationMaskDataset, custom_collate_fn
from torch.utils.data import DataLoader
from models.embedding.transformer_embedding import TransformerEmbedding, PositionalEncodingLoc, TransformerEmbeddingWithLoc
from models.blocks.encoder_layer import EncoderLayer
from models.blocks.decoder_layer import DecoderLayer
import math
import numpy as np
import torchvision.models as models
import torch.nn.functional as F


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, ifsigmoid=True):
        super().__init__()
        self.ifsigmoid = ifsigmoid
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.ifsigmoid:
            x = F.sigmoid(x)
        return x

class Encoder(nn.Module):

    def __init__(self, output_dim):
        super().__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(2048, output_dim)

    def forward(self, x):
        x = self.resnet(x)
        return x


class Decoder(nn.Module):
    def __init__(self, dec_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()

        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

        self.linear = nn.Linear(d_model, dec_voc_size)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        for layer in self.layers:
            trg = layer(trg, enc_src, trg_mask, src_mask)

        # pass to LM head
        output = self.linear(trg)
        return output


class TransformerMask(nn.Module):

    def __init__(self, d_model, n_head, max_len,
                 ffn_hidden, n_layers, drop_prob, device, hid_dim=4096, add_mapping=False, mode="cls",
                 share_loc_embed=False, attn_with_pos=False, num_bins=64
                 ):
        super().__init__()
        self.device = device
        self.max_len = max_len
        self.encoder = Encoder(output_dim=d_model)

        self.decoder = Decoder(d_model=d_model,
                               dec_voc_size=num_bins**2 + 5,
                               n_head=n_head,
                               max_len=max_len,
                               ffn_hidden=ffn_hidden,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device)
        self.share_loc_embed = share_loc_embed
        self.add_mapping = add_mapping
        self.mode = mode
        self.pos_emb = PositionalEncodingLoc(d_model, max_len+2, device)
        # cls 0, bos 1, eos 2, sep 3, pad 4, the rest follow.
        self.emb_dec = nn.Embedding(num_bins**2 + 5, d_model, padding_idx=4)
        if self.add_mapping:
            self.hid_map_encoder = MLP(d_model, 2*d_model, hid_dim, 2) #nn.Linear(d_model, hid_dim)
            self.hid_map_decoder = MLP(hid_dim, 2*d_model, d_model, 2) # nn.Linear(hid_dim, d_model)
        self.loc_to_locid, self.locid_to_loc = dict(), dict()


    def convert_to_grid(self, points, grid_size):
        grid_numbers = np.zeros(points.shape[0], dtype=int)
        for i, point in enumerate(points):
            x, y = point
            grid_x = min(int(x * grid_size), grid_size-1)
            grid_y = min(int(y * grid_size), grid_size-1)
            grid_number = grid_x + grid_y * grid_size
            grid_numbers[i] = grid_number
        return grid_numbers

    def convert_to_continuous(self, grid_numbers, grid_size):
        points = np.zeros((grid_numbers.shape[0], 2))
        for i, grid_number in enumerate(grid_numbers):
            grid_y = grid_number % grid_size
            grid_x = grid_number // grid_size
            x = (grid_x + 0.5) / grid_size
            y = (grid_y + 0.5) / grid_size
            points[i] = [x, y]
        return np.array(points)

    def forward(self, mask_pixel=None, input_dec=None, mask_dec=None, return_embedding=False):
        hidden_repr_enc = self.encoder(mask_pixel)
        if self.add_mapping:
            hidden_repr = self.hid_map_encoder(hidden_repr_enc)
            hidden_repr_dec = self.hid_map_decoder(hidden_repr)
        else:
            hidden_repr, hidden_repr_dec = hidden_repr_enc, hidden_repr_enc
        if return_embedding:
            return hidden_repr
        trg_mask = self.make_trg_mask(mask_dec)
        locations_embed_dec = self.emb_dec(input_dec)
        locations_embed_dec = locations_embed_dec * math.sqrt(locations_embed_dec.size(-1))
        pos_dec = self.pos_emb(locations_embed_dec)
        locations_embed_dec=  locations_embed_dec+pos_dec
        output = self.decoder(locations_embed_dec, hidden_repr_dec.unsqueeze(dim=1), trg_mask=trg_mask, src_mask=None)
        return {
            "output": output,
            "hidden_repr": hidden_repr
        }

    def make_src_mask(self, mask):
        src_mask = mask.unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, mask):
        trg_len = mask.shape[1]
        trg_pad_mask = mask.unsqueeze(1).unsqueeze(2)
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).to(self.device)
        trg_mask = trg_pad_mask * trg_sub_mask
        return trg_mask

    def make_trg_mask_inference(self):
        trg_sub_mask = torch.tril(torch.ones(self.max_len, self.max_len)).to(self.device)
        trg_mask = trg_sub_mask
        return trg_mask

    def generate(self, mask_pixel = None, encoder_repr = None):
        # src_mask = self.make_src_mask(src)
        # trg_mask = self.make_trg_mask(trg)
        device = next(self.parameters()).device
        assert (mask_pixel is not None or encoder_repr is not None)
        if mask_pixel is not None and encoder_repr is None:
            batch_size = mask_pixel.size(0)
            hidden_repr_enc = self.encoder(mask_pixel)
        else:
            batch_size = encoder_repr.size(0)
            hidden_repr_enc = encoder_repr
        if self.add_mapping:
            hidden_repr = self.hid_map_encoder(hidden_repr_enc)
            hidden_repr_dec = self.hid_map_decoder(hidden_repr)
        else:
            hidden_repr, hidden_repr_dec = hidden_repr_enc, hidden_repr_enc
        hidden_repr_dec = hidden_repr_dec.unsqueeze(1)
        dec_input_locations = torch.zeros((batch_size, self.max_len), dtype=torch.long).to(device)
        dec_input_locations[:, 0] = 1 # reserve one for <bos>
        dec_output_locations = torch.zeros((batch_size, self.max_len), dtype=torch.long).to(device)
        is_finished = torch.zeros(batch_size, dtype=torch.bool).to(device)
        finish_round = torch.zeros(batch_size, dtype=torch.int).to(device)
        last_index = torch.zeros([batch_size], dtype=torch.int).to(device)
        for i in range(self.max_len - 1):
            batch_index = torch.arange(batch_size).to(device)
            trg_mask = self.make_trg_mask_inference()
            locations_embed_dec = self.emb_dec(dec_input_locations)
            locations_embed_dec = locations_embed_dec * math.sqrt(locations_embed_dec.size(-1))
            pos_dec = self.pos_emb(locations_embed_dec)
            locations_embed_dec = locations_embed_dec+pos_dec
            cls_output = self.decoder(locations_embed_dec, hidden_repr_dec, trg_mask=trg_mask,
                                      src_mask=None)
            class_token = cls_output[batch_index, last_index, :].argmax(dim=-1)
            dec_output_locations[batch_index[~is_finished], last_index[~is_finished]] = class_token[~is_finished]
            # class_token = input_locations[batch_index, last_index+1]
            dec_input_locations[batch_index[~is_finished], last_index[~is_finished]+1] = class_token[~is_finished]
            is_eos_token = (class_token == 2).squeeze(dim=-1)
            newly_finished = (~is_finished) & (is_eos_token)
            if sum(newly_finished) > 0:
                finish_round[newly_finished] = i
            is_finished = is_finished | is_eos_token
            # Stop generating new points if <eos> token is generated for all sequences
            if is_finished.all():
                break
            # Save generated output if token is a location token
            # update last index
            last_index[~is_finished] += 1
        trimmed_outputs = []
        for round, output in zip(finish_round, dec_output_locations):
            if round != 0:
                valid_output = output[:round].detach().cpu().numpy()
            else:
                valid_output = output.detach().cpu().numpy()
            split_indices = np.where(valid_output == 3)[0]
            sub_arrays = np.split(valid_output, split_indices + 1)
            sub_arrays = [sub_array[sub_array != 3] for sub_array in sub_arrays]
            trimmed_outputs.append([self.convert_to_continuous(x-5 , 100) for x in sub_arrays])
        return trimmed_outputs


if __name__ == "__main__":
    import numpy as np
    from focal_loss import focal_loss
    import os
    import torch.nn.functional as F
    import torch

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    import matplotlib.pyplot as plt


    def visualize_polygons(gt_polygons, pred_polygons, masks):
        # fig, axs = plt.subplots(3, 6, figsize=(10, 10))
        fig, axs = plt.subplots(3, 3, figsize=(10, 10))

        for i, (gt_polygon, pred_polygon, mask) in enumerate(zip(gt_polygons, pred_polygons, masks)):
            if i >= 9:
                break
            # ax = axs[i // 3, i % 3 + (i % 3) * 2-1 ]
            ax = axs[i // 3, i % 3]
            for poly in gt_polygon:
                # ax.plot(poly[:, 0], poly[:, 1], "o", color='red', label='Ground Truth')
                # ax.plot([poly[0, 0], poly[-1, 0]], [poly[0, 1], poly[-1, 1]], "o", color='red')
                ax.plot(poly[:, 0], poly[:, 1], color='red', label='Ground Truth')
                ax.plot([poly[0, 0], poly[-1, 0]], [poly[0, 1], poly[-1, 1]], color='red')
            for poly in pred_polygon:
                # ax.plot(poly[:, 0], poly[:, 1], "o", color='blue', label='Prediction')
                # ax.plot([poly[0, 0], poly[-1, 0]], [poly[0, 1], poly[-1, 1]], "o", color='blue')
                ax.plot(poly[:, 0], poly[:, 1], color='blue', label='Prediction')
                ax.plot([poly[0, 0], poly[-1, 0]], [poly[0, 1], poly[-1, 1]], color='blue')

            # Set x and y axis limits to [0, 1]
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])

            # Connect the first and last point of each polygon

            ax.legend()
            # ax = axs[i // 3, i % 3 + (i % 3) * 2]
            # # plot masks
            # ax.imshow(mask, cmap='gray', origin='lower', alpha=0.3)

        plt.tight_layout()
        plt.show()


    torch.cuda.manual_seed(0)
    model = TransformerMask(
        d_model=512,
        max_len=1000,
        ffn_hidden=512,
        n_head=8,
        n_layers=6,
        add_mapping=False,
        num_bins=100,
        mode="cls",
        drop_prob=0.1,
        share_loc_embed=True,
        device="cuda")
    model.load_state_dict(torch.load(
        f"/home/pirenjie/transformer-master/saved/pixelmask_lr5e-4_200epoch_upper32_lower32_maxseg10_noaddmap_transform_continue-model.pt"))
    model = model.cuda()
    model.eval()
    device = model.device
    # val_dataset = SegmentationMaskDataset(path="/home/pirenjie/data/coco/annotations/instances_val2017.json", mask_path="/home/pirenjie/data/coco/masks/val",
    #                                   transformation=False, num_bins=100, downsample_num_lower=32, downsample_num_upper=32)
    val_dataset = SegmentationMaskDataset(path="/home/pirenjie/data/coco/annotations/instances_train2017.json", mask_path="/home/pirenjie/data/coco/masks/train",
                                          transformation=True, num_bins=100, downsample_num_lower=32, downsample_num_upper=32, size=100, transform_prob=0.5)
    val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=custom_collate_fn, num_workers=0)
    gts, preds, targets_token, masks = [], [], [], []
    start_id = 20
    with torch.no_grad():
        for i, (inputs_dec, mask_pixel, targets, masks_dec) in enumerate(val_loader):
            target_ = targets[0].cpu().numpy()[:-1]
            sub_arrays = np.split(target_, np.where(target_ == 3)[0] + 1)
            sub_arrays = [sub_array[sub_array != 3] for sub_array in sub_arrays]
            if len(sub_arrays) > 1:
                if i >= start_id:
                    gts.append([model.convert_to_continuous(gt-5 , 100) for gt in sub_arrays])
                    inputs_dec, mask_pixel, targets, masks_dec = \
                        inputs_dec.to(device), mask_pixel.to(device), targets.to(device), masks_dec.to(device)
                    # calculate cls and reg losses
                    outputs = model(inputs_dec, mask_pixel, masks_dec)
                    cls_pred = outputs["output"]
                    hidden_repr = outputs["hidden_repr"]
                    # cls_target = torch.zeros([len(cls_pred), 3]).to(device)
                    # cls_target[:, 0], cls_target[:, 1] = 0, 0
                    # cls_target[:, 2] = 1
                    cls_pred_reshape = cls_pred.contiguous().view(-1, cls_pred.shape[-1])
                    cls_target = targets.contiguous().view(-1).to(torch.long)

                    loss = F.cross_entropy(cls_pred_reshape[cls_target != -1], cls_target[cls_target != -1])
                    print(f"loss {loss}")
                    print(f"box pred {cls_pred[0].argmax(-1)[targets[0] != -1].flatten()}")
                    print(f"box gt {targets[0][targets[0] != -1].flatten()}")
                    generate_output = model.generate(mask_pixel)
                    targets_token.append(targets[0].cpu().numpy())
                    preds.append(generate_output[0])
                    masks.append(mask_pixel[0].squeeze().cpu().numpy())
            if len(gts) > 9:
                break
    visualize_polygons(gts, preds, masks)
