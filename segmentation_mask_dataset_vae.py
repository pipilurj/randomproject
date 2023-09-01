import copy
import os.path
import random

import numpy
import torch
from torch.utils.data import Dataset
import json
import numpy as np
from skimage import draw
from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader
from math import ceil, floor
import math
import torchvision.transforms as transforms

from PIL import Image
from dice_loss import dice_loss

def downsample_polygon(polygon, ds_num=32):
    points = np.array(polygon).reshape(int(len(polygon) / 2), 2)
    ds_rate = max(1, floor(len(points) // (ds_num - 1)))
    points = points[::ds_rate]
    return list(points[:ds_num].flatten())


def check_length(polygons):
    length = 0
    for polygon in polygons:
        length += len(polygon)
    return length


def approximate_polygon(poly, tolerance=2):
    poly = np.array(poly).reshape(int(len(poly) / 2), 2)
    new_poly = [poly[0]]
    for i in range(1, len(poly)):
        x1, y1 = new_poly[-1]
        x2, y2 = poly[i]
        dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        if dist > tolerance:
            new_poly.append(poly[i])
    new_poly = np.array(new_poly)
    return list(new_poly.flatten())


def approximate_polygons(polys, tolerance=1.0, max_length=400):
    tol = tolerance
    while check_length(polys) > max_length:
        polys_new = []
        for poly in polys:
            polys_new.append(approximate_polygon(poly, tolerance=tol))
        polys = polys_new
        tol += 2.0
    return polys


def interpolate_points(ps, pe):
    xs, ys = ps
    xe, ye = pe
    points = []
    dx = xe - xs
    dy = ye - ys
    if dx != 0:
        scale = dy / dx
        if xe > xs:
            x_interpolated = list(range(ceil(xs), floor(xe) + 1))
        else:
            x_interpolated = list(range(floor(xs), ceil(xe) - 1, -1))
        for x in x_interpolated:
            y = ys + (x - xs) * scale
            points.append([x, y])
    if dy != 0:
        scale = dx / dy
        if ye > ys:
            y_interpolated = list(range(ceil(ys), floor(ye) + 1))
        else:
            y_interpolated = list(range(floor(ys), ceil(ye) - 1, -1))
        for y in y_interpolated:
            x = xs + (y - ys) * scale
            points.append([x, y])
    if xe > xs:
        points = sorted(points, key=lambda x: x[0])
    else:
        points = sorted(points, key=lambda x: -x[0])
    return points


def prune_points(points, th=0.1):
    points_pruned = [points[0]]
    for i in range(1, len(points)):
        x1, y1 = points_pruned[-1]
        x2, y2 = points[i]
        dist = (x2 - x1) ** 2 + (y2 - y1) ** 2
        if dist > th:
            points_pruned.append(points[i])
    return points_pruned


def interpolate_polygon(polygon):
    points = np.array(polygon).reshape(int(len(polygon) / 2), 2)
    # points = polygon
    points_interpolated = []
    points_interpolated.append(points[0])
    for i in range(0, len(points) - 1):
        points_i = interpolate_points(points[i], points[i + 1])
        points_interpolated += points_i
        points_interpolated.append(points[i + 1])
    points_interpolated = prune_points(points_interpolated)
    polygon_interpolated = np.array(points_interpolated)
    return list(polygon_interpolated.flatten())


def interpolate_polygons(polygons):
    polygons_i = []
    for polygon in polygons:
        polygons_i.append(interpolate_polygon(polygon))
    return polygons_i


def downsample_polygons(polygons, ds_num=25):
    polygons_ds = []
    for polygon in polygons:
        polygons_ds.append(downsample_polygon(polygon, ds_num))
    return polygons_ds


def revert_direction(poly):
    poly = np.array(poly).reshape(int(len(poly) / 2), 2)
    poly = poly[::-1, :]
    return list(poly.flatten())


def reorder_points(poly):
    poly = np.array(poly)
    xs = poly[::2]
    ys = poly[1::2]
    points = np.array(poly).reshape(int(len(poly) / 2), 2)
    start = np.argmin(xs ** 2 + ys ** 2)  # smallest distance to the origin
    poly_reordered = np.concatenate([points[start:], points[:start]], 0)
    return list(poly_reordered.flatten())


def is_clockwise(poly):
    n = len(poly) // 2
    xs = poly[::2]
    xs.append(xs[0])
    ys = poly[1::2]
    ys.append(ys[0])
    area = 0
    for i in range(n):
        x1, y1 = xs[i], ys[i]
        x2, y2 = xs[i + 1], ys[i + 1]
        area += (x2 - x1) * (y2 + y1)
    return area < 0


def close_polygon_contour(poly):
    poly = np.array(poly).reshape(int(len(poly) / 2), 2)
    x1, y1 = poly[0]
    x2, y2 = poly[-1]
    if x1 != x2:
        poly = np.concatenate([poly, [poly[0]]], 0)
    return list(poly.flatten())


def close_polygons_contour(polygons):
    polygons_closed = []
    for polygon in polygons:
        polygon_closed = close_polygon_contour(polygon)
        polygons_closed.append(polygon_closed)
    return polygons_closed


def _computeIoU(mask1, mask2):
    I = np.sum(np.logical_and(mask1, mask2))
    U = np.sum(np.logical_or(mask1, mask2))
    return I, U


def getIoU(mask1, mask2):
    I, U = _computeIoU(mask1, mask2)
    if U == 0:
        return 0
    else:
        return I / U


def convert_pts(coeffs):
    pts = []
    for i in range(len(coeffs) // 2):
        pts.append([coeffs[2 * i + 1], coeffs[2 * i]])  # y, x
    return np.array(pts, np.int32)


def get_mask_from_codes(codes, img_size):
    masks = [np.zeros(img_size)]
    for code in codes:
        if len(code) > 0:
            try:
                mask = draw.polygon2mask(img_size, convert_pts(code))
                mask = np.array(mask, np.uint8)
            except:
                mask = np.zeros(img_size)
            masks.append(mask)
    mask = sum(masks)
    mask = mask > 0
    return mask.astype(np.uint8)


def process_polygons(polygons, redirection=True, reorder=True, close=False):
    polygons_processed = []
    for polygon in polygons:
        if redirection and not is_clockwise(polygon):
            polygon = revert_direction(polygon)
        if reorder:
            polygon = reorder_points(polygon)
        if close:
            polygon = close_polygon_contour(polygon)
        # polygons_processed.append(sorted(polygon, key=lambda x: (x[0] ** 2 + x[1] ** 2, x[0], x[1])))
        polygons_processed.append(polygon)
    polygons_processed = sorted(polygons_processed, key=lambda x: (x[0] ** 2 + x[1] ** 2, x[0], x[1]))
    return polygons_processed


def process_polygon(polygon, redirection=True, reorder=True, close=False):
    if redirection and not is_clockwise(polygon):
        polygon = revert_direction(polygon)
    if reorder:
        polygon = reorder_points(polygon)
    if close:
        polygon = close_polygon_contour(polygon)
    # polygons = sorted(polygons_processed, key=lambda x: (x[0] ** 2 + x[1] ** 2, x[0], x[1]))
    return np.stack(polygon)


def get_padded_sample_target_mask(samples):
    # <cls> 0, <bos> 1, <eos> 2, <sep> 3, <pad> 4
    max_len = max([sum(len(seg) for seg in segments) + len(segments) - 1 for segments in samples])  # length with <sep>
    inputs_dec_all, targets_all, masks_dec_all = [], [], []
    for sample in samples:
        input_dec = np.array([1])  # <bos>
        for i, segment in enumerate(sample):
            segment = segment + 5
            input_dec = np.concatenate((input_dec, segment), axis=0)
            if i < len(sample) - 1:
                input_dec = np.concatenate((input_dec, np.array([3])), axis=0)
        target = np.concatenate((input_dec[1:], np.array([2])), axis=0)  # <eos>
        mask_dec = np.array([1 for _ in range(len(input_dec))])  # one additional mask for bos
        # while len(input_dec) < max_len + 1:
        remainder = max_len + 1 - len(input_dec)
        input_dec = np.concatenate((input_dec, np.array([4] * remainder)), axis=0)
        target = np.concatenate((target, np.array([-1] * remainder)), axis=0)
        mask_dec = np.concatenate((mask_dec, np.array([0] * remainder)), axis=0)
        inputs_dec_all.append(input_dec)
        targets_all.append(target)
        masks_dec_all.append(mask_dec)
        # break
    inputs_dec_all, targets_all, masks_dec_all = numpy.stack(inputs_dec_all), numpy.stack(targets_all), numpy.array(
        masks_dec_all)
    return torch.tensor(inputs_dec_all, dtype=torch.long), torch.tensor(targets_all, dtype=torch.long), torch.tensor(
        masks_dec_all, dtype=torch.float32)


def apply_translation(polygon, tx, ty):
    # Apply translation transformation to the polygon
    translated_polygon = polygon + np.array([tx, ty])
    return translated_polygon


def apply_shearing(polygon, sx, sy):
    # Apply shearing transformation to the polygon
    sheared_polygon = polygon + np.array([polygon[:, 1] * sx, polygon[:, 0] * sy]).T
    return sheared_polygon


# def transform(polygons, w, h, margin=0.5, translation_range = [0, 5]):
#     translation_range = translation_range
#     shearing_range = [-0.5, 0.5]
#     transformed_polygons = []
#     for polygon in polygons:
#         polygon = np.array(polygon).reshape(-1, 2)
#         min_x, max_x, min_y, max_y = polygon[:, 0].min(), polygon[:, 0].max(), polygon[:, 1].min(), polygon[:, 1].max()
#         margin_x, margin_y = margin*(max_x - min_x), margin*(max_y - min_y)
#         transx, transy = random.uniform(-min((max_x-margin_x), random.uniform(*translation_range)), min((w-min_x-margin_x), random.uniform(*translation_range))), random.uniform(-min((max_y-margin_y), random.uniform(*translation_range)), min((h-min_y-margin_y), random.uniform(*translation_range)))
#         polygon = apply_translation(polygon, transx, transy)
#         # polygon = apply_shearing(polygon, shearx, sheary)
#         polygon = process_polygon(list(polygon.flatten()))
#         polygon = polygon.reshape(-1, 2)
#         polygon[:, 0] = np.clip(polygon[:, 0], 0, w)
#         polygon[:, 1] = np.clip(polygon[:, 1], 0, h)
#         transformed_polygons.append(polygon.flatten().tolist())
#
#     return transformed_polygons
def transform(polygons, w, h, margin=0.5, translations = [0, 0]):
    transx, transy = translations
    shearing_range = [-0.5, 0.5]
    transformed_polygons = []
    for polygon in polygons:
        polygon = np.array(polygon).reshape(-1, 2)
        polygon = apply_translation(polygon, transx, transy)
        # polygon = apply_shearing(polygon, shearx, sheary)
        polygon = process_polygon(list(polygon.flatten()))
        polygon = polygon.reshape(-1, 2)
        polygon[:, 0] = np.clip(polygon[:, 0], 0, w)
        polygon[:, 1] = np.clip(polygon[:, 1], 0, h)
        transformed_polygons.append(polygon.flatten().tolist())

    return transformed_polygons

def custom_collate_fn(samples):
    # Implement your logic to handle samples of different lengths
    # token id: padding: -1, loc:
    # len(target): max_len + 1; len(mask): max_len + 1 ; len(sample): max_len
    segment_discrete, mask_pixel = [s[0] for s in samples], [s[1] for s in
                                                             samples]  # Process the samples to make them of the same length or format
    inputs_dec_all, targets_all, masks_dec_all = get_padded_sample_target_mask(
        segment_discrete)
    return inputs_dec_all, torch.stack(mask_pixel), targets_all, masks_dec_all


def scale_segments(segments, scale):
    segments_scaled = []
    for segment in segments:
        n_point = len(segment) // 2
        scale_anchor = np.concatenate([np.array(scale) for _ in range(n_point)], 0)
        segments_scaled.append(segment / scale_anchor)
    return segments_scaled


def get_pos_neg(anchor, pos, neg, scale_anchor, scale_positive, scale_negative):
    mask_anchor = get_mask_from_codes([anchor], scale_anchor)
    mask_pos = get_mask_from_codes([pos], scale_positive)
    mask_neg = get_mask_from_codes([neg], scale_negative)

    iou_positive = getIoU(mask_anchor, mask_pos)
    iou_negative = getIoU(mask_anchor, mask_neg)

    # Determine positive and negative based on IoU
    if iou_positive > iou_negative:
        return pos, neg
    else:
        return neg, pos


class SegmentationMaskDataset(Dataset):
    # class label for tokens:
    # loc: 0; sep: 1; eos: 2
    def __init__(self, path="/home/pirenjie/data/coco/annotations/instances_train2017.json",
                 mask_path="/home/pirenjie/data/coco/masks/train", downsample_num_upper=32,
                 downsample_num_lower=32, transformation=False, size=None, num_bins=64, max_segments=15,
                 transform_prob=0):
        self.coco = COCO(path)
        annotations = self.coco.dataset["annotations"]
        self.legal_indices = list(range(len(annotations)))
        if size is not None:
            self.legal_indices = list(range(size))
        # self.legal_indices = list(range(20))
        self.downsample_num_upper = downsample_num_upper
        self.downsample_num_lower = downsample_num_lower
        self.num_bins = num_bins
        self.transformation = transformation
        self.loc_to_locid, self.locid_to_loc = dict(), dict()
        self.transform_prob = transform_prob
        num = 0
        for x in range(0, num_bins):
            for y in range(0, num_bins):
                self.loc_to_locid.update({f"<bin_{x}_{y}>": num})
                num += 1
        self.legal_indices = [i for i in self.legal_indices if isinstance(annotations[i]["segmentation"], list) and len(
            annotations[i]["segmentation"]) <= max_segments]
        print(f"valid points {len(self.legal_indices)}")
        self.mask_path = mask_path
        self.mask_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256), antialias=True)
        ])
        self.mask_perturbation_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256), antialias=True),
            transforms.RandomAffine(degrees=10, translate=(0.3, 0.3), scale=(0.9, 1.1), shear=0.1),
        ])
    def __len__(self):
        return len(self.legal_indices)

    def convert_index(self, index):
        return self.legal_indices[index]

    def transform_mask(self, segments_processed, w, h, anno, range):
        segments = transform(segments_processed, w, h, translation_range=range)
        segments_processed = process_polygons(segments)
        anno = copy.deepcopy(anno)
        anno["segmentation"] = segments_processed
        mask_pixel = self.coco.annToMask(anno)
        mask_pixel = np.array(mask_pixel).astype(np.float32)
        mask_pixel = self.mask_transform(mask_pixel)
        return mask_pixel

    def load_segment_scale(self, index):
        # debug
        real_index = self.convert_index(index)
        anno = self.coco.dataset["annotations"][real_index]
        img_id = anno["image_id"]
        img_info = self.coco.loadImgs([img_id])[0]
        w, h = img_info["width"], img_info["height"]
        segments = anno["segmentation"]
        segments_processed = process_polygons(segments)
        if self.transformation and random.uniform(0, 1) < self.transform_prob:
            segments_processed = transform(segments_processed, w, h)
            segments_processed = process_polygons(segments_processed)
            anno["segmentation"] = segments_processed
            mask_pixel = self.coco.annToMask(anno)
            mask_pixel = np.array(mask_pixel).astype(np.float32)
        else:
            mask_pixel = np.array(Image.open(os.path.join(self.mask_path, f"mask_{real_index}.png")).convert("1")).astype(np.float32)
        # positive:
        mask_pixel_anchor = self.mask_transform(mask_pixel)
        # translation_pos = [random.uniform(*[-10, 10]), random.uniform(*[-10, 10])]
        # translation_neg = [random.uniform(*[-10, 10]), random.uniform(*[-10, 10])]
        # mask_pixel_pos = self.transform_mask(segments_processed, w, h, anno, [0,5])
        # mask_pixel_neg = self.transform_mask(segments_processed, w, h, anno, [6,10])
        mask_pixel_pos = self.mask_perturbation_transform(mask_pixel)
        mask_pixel_neg = self.mask_perturbation_transform(mask_pixel)
        dice_pos = dice_loss(mask_pixel_anchor.float(), mask_pixel_pos.float(), multiclass=False)
        dice_neg = dice_loss(mask_pixel_anchor.float(), mask_pixel_neg.float(), multiclass=False)
        if dice_neg < dice_pos:
            return mask_pixel_anchor, mask_pixel_neg, mask_pixel_pos
        else:
            return mask_pixel_anchor, mask_pixel_pos, mask_pixel_neg

    def quantize_polygon(self, polygons):
        quantized_polygons = []
        for polygon in polygons:
            polygon = polygon.reshape(-1, 2)
            quant_poly = polygon * self.num_bins
            discrete_points = []
            for p in quant_poly:
                discrete_points.append(self.loc_to_locid[
                                           f"<bin_{min(math.floor(p[0]), self.num_bins - 1)}_{min(math.floor(p[1]), self.num_bins - 1)}>"])
            quantized_polygons.append(np.array(discrete_points, dtype="i"))
        return quantized_polygons

    def __getitem__(self, index):
        mask_pixel, mask_pixel_pos, mask_pixel_neg = self.load_segment_scale(index)

        return mask_pixel, mask_pixel_pos, mask_pixel_neg


if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt


    def convert_to_continuous(grid_numbers, grid_size):
        points = np.zeros((grid_numbers.shape[0], 2))
        for i, grid_number in enumerate(grid_numbers):
            grid_y = grid_number % grid_size
            grid_x = grid_number // grid_size
            x = (grid_x + 0.5) / grid_size
            y = (grid_y + 0.5) / grid_size
            points[i] = [x, y]
        return np.array(points)


    def visualize_polygons(masks, pred_polygons):
        fig, axs = plt.subplots(3, 6, figsize=(20, 10))

        for i, (mask, pred_polygon) in enumerate(zip(masks, pred_polygons)):
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


    seg_dataset = SegmentationMaskDataset("/home/pirenjie/data/coco/annotations/instances_train2017.json",
                                          transformation=True, num_bins=100, transform_prob=1)
    loader = DataLoader(seg_dataset, batch_size=100)
    # time_start = time.time()
    # for inputs_dec, mask_pixel, targets, masks_dec in loader:
    #     time_end = time.time()
    #     print(f"time spent : {time_end - time_start}")
    #     time_start = time.time()
    segments, masks = [], []
    start = 10
    for i, (mask1, mask2, mask3) in enumerate(loader):
        pass
    #     if len(masks) >= 9:
    #         break
    #     if i >= start:
    #         target_ = targets[0].cpu().numpy()[:-1]
    #         sub_arrays = np.split(target_, np.where(target_ == 3)[0] + 1)
    #         sub_arrays = [sub_array[sub_array != 3] for sub_array in sub_arrays]
    #         segments.append([convert_to_continuous(gt - 5, 100) for gt in sub_arrays])
    #         masks.append(mask.squeeze())
    # visualize_polygons(masks, segments)
