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
        # if redirection and not is_clockwise(polygon):
        #     polygon = revert_direction(polygon)
        if reorder:
            polygon = reorder_points(polygon)
        if close:
            polygon = close_polygon_contour(polygon)
        # polygons_processed.append(sorted(polygon, key=lambda x: (x[0] ** 2 + x[1] ** 2, x[0], x[1])))
        polygons_processed.append(polygon)
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
    # <cls> 0, <bos> 1, <eos> 2, <pad> 3
    max_len = max([len(s) for s in samples])
    inputs_enc_all, inputs_dec_all = [], []
    targets_all, masks_enc_all, masks_dec_all = [], [], []
    for sample in samples:
        input = copy.deepcopy(sample) + 4
        input_enc = np.concatenate((np.array([0]), input, np.array([2])), axis=0)  # concat <cls> and <eos>
        input_dec = np.concatenate((np.array([1]), input), axis=0)  # concat <bos>
        target = np.concatenate((input, np.array([2])), axis=0)  # concat <eos>
        mask_enc = [1 for _ in range(len(input) + 2)]  # one additional mask for cls, another for <eos>
        mask_dec = [1 for _ in range(len(input) + 1)]  # one additional mask for bos
        while len(input_enc) < max_len + 2:
            input_enc = np.concatenate((input_enc, np.array([3])), axis=0)
            input_dec = np.concatenate((input_dec, np.array([3])), axis=0)
            target = np.concatenate((target, [-1]), axis=0)
            mask_enc.append(0)
            mask_dec.append(0)
        inputs_enc_all.append(input_enc)
        inputs_dec_all.append(input_dec)
        targets_all.append(target)
        masks_enc_all.append(mask_enc)
        masks_dec_all.append(mask_dec)
        # break
    inputs_enc_all, inputs_dec_all, targets_all, masks_enc_all, masks_dec_all = numpy.stack(
        inputs_enc_all), numpy.stack(inputs_dec_all), numpy.stack(targets_all), numpy.array(
        masks_enc_all), numpy.array(
        masks_dec_all)
    return torch.tensor(inputs_enc_all, dtype=torch.long), torch.tensor(inputs_dec_all, dtype=torch.long), torch.tensor(targets_all, dtype=torch.long), torch.tensor(
        masks_enc_all, dtype=torch.float32), torch.tensor(masks_dec_all, dtype=torch.float32)


def apply_translation(polygon, tx, ty):
    # Apply translation transformation to the polygon
    translated_polygon = polygon + np.array([tx, ty])
    return translated_polygon


def apply_shearing(polygon, sx, sy):
    # Apply shearing transformation to the polygon
    sheared_polygon = polygon + np.array([polygon[:, 1] * sx, polygon[:, 0] * sy]).T
    return sheared_polygon


def transform(polygon):
    translation_range = [-0.1, 0.1]
    shearing_range = [-0.1, 0.1]
    polygon = apply_translation(polygon, random.uniform(*translation_range), random.uniform(*translation_range))
    polygon = apply_shearing(polygon, random.uniform(*shearing_range), random.uniform(*shearing_range))
    polygon = process_polygon(list(polygon.flatten()))
    polygon = np.clip(polygon, 0, 1)
    return polygon


def custom_collate_fn(samples):
    # Implement your logic to handle samples of different lengths
    # token id: padding: -1, loc:
    # len(target): max_len + 1; len(mask): max_len + 1 ; len(sample): max_len
    anchors, positive, negative = [s[0] for s in samples], [s[1] for s in samples], [s[2] for s in
                                                                                     samples]  # Process the samples to make them of the same length or format
    anchor_input_enc_all, anchor_input_dec_all, anchor_targets, anchor_masks_enc, anchor_masks_dec = get_padded_sample_target_mask(
        anchors)
    positive_input_enc_all, positive_input_dec_all, positive_targets,  positive_masks_enc, positive_masks_dec = get_padded_sample_target_mask(
        positive)
    negative_input_enc_all, negative_input_dec_all, negative_targets, negative_masks_enc, negative_masks_dec = get_padded_sample_target_mask(
        negative)
    return anchor_input_enc_all, anchor_input_dec_all, anchor_targets, anchor_masks_enc, anchor_masks_dec, positive_input_enc_all, positive_input_dec_all, positive_targets, positive_masks_enc, positive_masks_dec, negative_input_enc_all, negative_input_dec_all, negative_targets, negative_masks_enc, negative_masks_dec


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


class SegmentationDiscreteDataset(Dataset):
    # class label for tokens:
    # loc: 0; sep: 1; eos: 2
    def __init__(self, path="/home/pirenjie/data/coco/annotations/instances_train2017.json", downsample_num_upper=64,
                 downsample_num_lower=32, transformation=True, size=None, num_bins=64):
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
        num = 0
        for x in range(0, num_bins):
            for y in range(0, num_bins):
                self.loc_to_locid.update({f"<bin_{x}_{y}>": num})
                num += 1
        self.legal_indices = [i for i in self.legal_indices if isinstance(annotations[i]["segmentation"], list)]
        print(f"valid points {len(self.legal_indices)}")

    def __len__(self):
        return len(self.legal_indices)

    def convert_index(self, index):
        return self.legal_indices[index]

    def load_segment_scale(self, index):
        # debug
        anno = self.coco.dataset["annotations"][index]
        img_id = anno["image_id"]
        img_info = self.coco.loadImgs([img_id])[0]
        w, h = img_info["width"], img_info["height"]
        masks = anno["segmentation"]
        # Iterate over each subsegment in the mask
        masks_processed = process_polygons(masks)
        masks_interpolated = interpolate_polygons(masks_processed)
        selected_segment = random.choice(masks_interpolated)
        ds_num1, ds_num2 = random.randint(self.downsample_num_lower, self.downsample_num_upper), random.randint(
            self.downsample_num_lower, self.downsample_num_upper)
        # debug
        masks_downsampled_1 = downsample_polygon(selected_segment, ds_num=ds_num1)
        masks_downsampled_2 = downsample_polygon(selected_segment, ds_num=ds_num2)
        segment_array_anchor = np.array(masks_downsampled_1).flatten()
        segment_array_pos = np.array(masks_downsampled_2).flatten()
        return np.clip(segment_array_anchor, 0, 10000), np.clip(segment_array_pos, 0, 10000), np.array([w, h])

    def quantize_polygon(self, polygon):
        polygon = polygon.reshape(-1, 2)
        quant_poly = polygon * self.num_bins
        discrete_points = []
        for p in quant_poly:
            discrete_points.append(self.loc_to_locid[f"<bin_{min(math.floor(p[0]), self.num_bins-1)}_{min(math.floor(p[1]), self.num_bins-1)}>"])
        return np.array(discrete_points, dtype="i")

    def __getitem__(self, index):
        index = self.convert_index(index)
        segment_array_anchor, segment_array_positive, scale = self.load_segment_scale(index)
        n_point = len(segment_array_anchor) // 2
        scale_anchor = np.concatenate([np.array(scale) for _ in range(n_point)], 0)
        segment_rescaled_anchor = segment_array_anchor / scale_anchor
        segment_array_negative, _, scale_neg = self.load_segment_scale(random.choice(self.legal_indices))
        scale_positive = np.concatenate([np.array(scale) for _ in range(len(segment_array_positive) // 2)], 0)
        scale_negative = np.concatenate([np.array(scale_neg) for _ in range(len(segment_array_negative) // 2)], 0)
        segment_rescaled_positive = segment_array_positive / scale_positive  # scale_positive[0]
        segment_rescaled_negative = segment_array_negative / scale_negative  # scale_negative[0]
        if self.transformation:
            segment_rescaled_anchor = transform(segment_rescaled_anchor.reshape(-1, 2))
            segment_rescaled_positive = transform(segment_rescaled_positive.reshape(-1, 2))

        anchor_discrete = self.quantize_polygon(
            segment_rescaled_anchor)
        pos_discrete = self.quantize_polygon(
            segment_rescaled_positive)
        neg_discrete = self.quantize_polygon(
            segment_rescaled_negative)

        return anchor_discrete, pos_discrete, neg_discrete


if __name__ == "__main__":
    import time

    seg_dataset = SegmentationDiscreteDataset("/home/pirenjie/data/coco/annotations/instances_train2017.json",
                                              transformation=False)
    loader = DataLoader(seg_dataset, batch_size=1000, collate_fn=custom_collate_fn)
    time_start = time.time()
    for anchor_input_enc_all, anchor_input_dec_all, anchor_targets, anchor_masks_enc, anchor_masks_dec, positive_input_enc_all, positive_input_dec_all, positive_targets, positive_masks_enc, positive_masks_dec, negative_input_enc_all, negative_input_dec_all, negative_targets, negative_masks_enc, negative_masks_dec in loader:
        time_end = time.time()
        print(f"time spent : {time_end - time_start}")
        time_start = time.time()
