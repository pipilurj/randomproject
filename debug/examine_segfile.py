import copy
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

from PIL import Image
def downsample_polygon(polygon, ds_num=32):
    points = np.array(polygon).reshape(int(len(polygon) / 2), 2)
    ds_rate = max(1, floor(len(points)//(ds_num-1)))
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
    # points = np.array(polygon).reshape(int(len(polygon) / 2), 2)
    points = polygon
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


def downsample_polygons(polygons, ds_rate=25):
    polygons_ds = []
    for polygon in polygons:
        polygons_ds.append(downsample_polygon(polygon, ds_rate))
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
    return poly_reordered


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
        polygons_processed.append(polygon)
        # polygons_processed.append(sorted(polygon, key=lambda x: (x[0] ** 2 + x[1] ** 2, x[0], x[1])))
    # polygons = sorted(polygons_processed, key=lambda x: (x[0] ** 2 + x[1] ** 2, x[0], x[1]))
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
    processed_samples = samples  # Process the samples to make them of the same length or format
    max_len = max([len(s) for s in processed_samples])
    targets_all, masks_enc_all, masks_dec_all, samples_extended, loc_flags_all = [], [], [], [], []
    for sample in processed_samples:
        sample_extended = copy.deepcopy(sample)
        sample_extended = np.concatenate((sample_extended, np.array([[0., 0.]])),
                                         axis=0)  # add one more for <eos> place holder
        target_loc = [0 for _ in range(len(sample))]  #
        target_loc.append(1)  # add eos token
        target_loc.append(-1)  # the extra <eos> token
        loc_flags = [1 for _ in range(len(sample))]  # 1 represents location token
        loc_flags.append(3)  # 3 represents <eos>
        mask_enc = [1 for _ in range(len(sample) + 2)]  # one additional mask for cls, another for <eos>
        mask_dec = [1 for _ in range(len(sample) + 1)]  # one additional mask for bos
        mask_dec.append(0)  # mask out <eos>
        while len(sample_extended) < max_len + 1:
            sample_extended = np.concatenate((sample_extended, np.array([[0., 0.]])), axis=0)
            target_loc.append(-1)
            mask_enc.append(0)
            mask_dec.append(0)
            loc_flags.append(0)
        samples_extended.append(sample_extended)
        targets_all.append(target_loc)
        masks_enc_all.append(mask_enc)
        masks_dec_all.append(mask_dec)
        loc_flags_all.append(loc_flags)
        if len(loc_flags) != len(sample_extended):
            pass
    samples_extended, targets_all, masks_enc_all, masks_dec_all, loc_flags_all = numpy.array(
        samples_extended), numpy.array(targets_all), numpy.array(masks_enc_all), numpy.array(
        masks_dec_all), numpy.array(loc_flags_all)
    return torch.tensor(samples_extended, dtype=torch.float32), torch.tensor(targets_all,
                                                                             dtype=torch.float32), torch.tensor(
        masks_enc_all, dtype=torch.float32), torch.tensor(masks_dec_all, dtype=torch.float32), torch.tensor(
        loc_flags_all, dtype=torch.float32)

def apply_translation(polygon, tx, ty):
    # Apply translation transformation to the polygon
    translated_polygon = polygon + np.array([tx, ty])
    return translated_polygon

def apply_shearing(polygon, sx, sy):
    # Apply shearing transformation to the polygon
    sheared_polygon = polygon + np.array([polygon[:, 1] * sx, polygon[:, 0] * sy]).T
    return sheared_polygon

def transform(polygon):
    translation_range = [-0.05, 0.05]
    shearing_range = [-0.05, 0.05]
    polygon = apply_translation(polygon, random.uniform(*translation_range), random.uniform(*translation_range))
    polygon = apply_shearing(polygon, random.uniform(*shearing_range), random.uniform(*shearing_range))
    polygon = np.clip(polygon, 0, 1)
    polygon = process_polygon(list(polygon.flatten()))
    return polygon

def custom_collate_fn(samples):
    # Implement your logic to handle samples of different lengths
    # token id: padding: -1, loc:
    # len(target): max_len + 1; len(mask): max_len + 1 ; len(sample): max_len
    anchors, positive, negative = [s[0] for s in samples], [s[1] for s in samples], [s[2] for s in
                                                                                     samples]  # Process the samples to make them of the same length or format
    anchor_samples, anchor_targets, anchor_masks_enc, anchor_masks_dec, anchor_locflags = get_padded_sample_target_mask(
        anchors)
    positive_samples, positive_targets, positive_masks_enc, positive_masks_dec, positive_locflags = get_padded_sample_target_mask(
        positive)
    negative_samples, negative_targets, negative_masks_enc, negative_masks_dec, negative_locflags = get_padded_sample_target_mask(
        negative)
    return anchor_samples, anchor_targets, anchor_masks_enc, anchor_masks_dec, anchor_locflags, positive_samples, positive_targets, positive_masks_enc, positive_masks_dec, positive_locflags, negative_samples, negative_targets, negative_masks_enc, negative_masks_dec, negative_locflags


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


class SegmentationDataset(Dataset):
    # class label for tokens:
    # loc: 0; sep: 1; eos: 2
    def __init__(self, path="/home/pirenjie/data/coco/annotations/instances_train2017.json", downsample_num_upper = 24, downsample_num_lower = 16, transformation = True):
        self.coco = COCO(path)
        annotations = self.coco.dataset["annotations"]
        self.legal_indices = list(range(len(annotations)))
        # self.legal_indices = list(range(20))
        self.downsample_num_upper = downsample_num_upper
        self.downsample_num_lower = downsample_num_lower
        self.transformation = transformation
        self.legal_indices = [i for i in self.legal_indices if isinstance(annotations[i]["segmentation"], list)]
        print(f"valid points {len(self.legal_indices)}")

    def __len__(self):
        return len(self.legal_indices)

    def convert_index(self, index):
        return self.legal_indices[index]

    def load_segment_scale(self, index):
        index = self.convert_index(index)
        anno = self.coco.dataset["annotations"][index]
        img_id = anno["image_id"]
        img_info = self.coco.loadImgs([img_id])[0]
        w, h = img_info["width"], img_info["height"]
        masks = anno["segmentation"]
        # Iterate over each subsegment in the mask
        masks_processed = process_polygons(masks)
        masks_interpolated = interpolate_polygons(masks_processed)
        selected_segment = random.choice(masks_interpolated)
        ds_num1, ds_num2 = random.randint(self.downsample_num_lower, self.downsample_num_upper), random.randint(self.downsample_num_lower, self.downsample_num_upper)
        masks_downsampled_1 = downsample_polygon(selected_segment, ds_num=ds_num1)
        masks_downsampled_2 = downsample_polygon(selected_segment, ds_num=ds_num2)
        # for segment in masks_processed:
        segment_array_anchor = np.array(masks_downsampled_1).flatten()
        segment_array_pos = np.array(masks_downsampled_2).flatten()
        return segment_array_anchor, segment_array_pos, np.array(selected_segment), np.array([w, h])

    def __getitem__(self, index):
        segment_array_anchor, segment_array_positive, segment_interpolated, scale = self.load_segment_scale(index)
        n_point = len(segment_array_anchor) // 2
        n_point_interpolated = len(segment_interpolated) // 2
        scale_anchor = np.concatenate([np.array(scale) for _ in range(n_point)], 0)
        scale_anchor_interpolated = np.concatenate([np.array(scale) for _ in range(n_point_interpolated)], 0)
        segment_rescaled_anchor = segment_array_anchor / scale_anchor
        segment_interpolated_interpolated = segment_interpolated / scale_anchor_interpolated
        segment_array_negative, _, _, _ = self.load_segment_scale(random.choice(list(range(len(self)))))
        # segment_array_positive, segment_array_negative = get_pos_neg(segment_array_anchor, segment_array_positive, segment_array_negative, scale_anchor[0], scale_positive[0], scale_negative[0])
        # segment_array_positive, segment_array_negative = get_pos_neg(segment_array_anchor, segment_array_positive,
        #                                                              segment_array_negative, scale, scale, scale)
        scale_positive = np.concatenate([np.array(scale) for _ in range(len(segment_array_positive) // 2)], 0)
        scale_negative = np.concatenate([np.array(scale) for _ in range(len(segment_array_negative) // 2)], 0)
        segment_rescaled_positive = segment_array_positive / scale_positive  # scale_positive[0]
        segment_rescaled_negative = segment_array_negative / scale_negative  # scale_negative[0]
        if self.transformation:
            segment_rescaled_anchor = transform(segment_rescaled_anchor.reshape(-1,2))
            segment_rescaled_positive = transform(segment_rescaled_positive.reshape(-1,2))

        # segments_updated.append(segment_rescaled.reshape(-1,2))
        # labels.extend([0]*n_point)
        # labels.append(1)

        return segment_rescaled_anchor.reshape(-1, 2), segment_rescaled_positive.reshape(-1,
                                                                                         2), segment_rescaled_negative.reshape(
            -1, 2), segment_interpolated_interpolated.reshape(-1, 2)  # , labels[:-1]


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    seg_dataset = SegmentationDataset("/home/pirenjie/data/coco/annotations/instances_val2017.json")
    polygons_anchor, polygons_pos = [], []
    for i in range(9):
        pol = seg_dataset[i][-1]
        pos = seg_dataset[i][1]
        polygons_anchor.append(pol)
        polygons_pos.append(pos)

    def visualize_polygons(polygons_anchor, polygons_pos):
        fig, axs = plt.subplots(3, 3, figsize=(10, 10))

        for i, (polygon_anchor, polygon_pos) in enumerate(zip(polygons_anchor, polygons_pos)):
            ax = axs[i // 3, i % 3]
            ax.plot(polygon_anchor[:, 0], polygon_anchor[:, 1], color='red', label='anchor')
            ax.plot(polygon_pos[:, 0], polygon_pos[:, 1], color='blue', label='pos')

            # Set x and y axis limits to [0, 1]
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])

            # Connect the first and last point of each polygon
            ax.plot([polygon_anchor[0, 0], polygon_anchor[-1, 0]], [polygon_anchor[0, 1], polygon_anchor[-1, 1]], color='red')
            ax.plot([polygon_pos[0, 0], polygon_pos[-1, 0]], [polygon_pos[0, 1], polygon_pos[-1, 1]], color='blue')

            ax.legend()

        plt.tight_layout()
        plt.show()

    visualize_polygons(polygons_anchor, polygons_pos)
