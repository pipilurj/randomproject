import torch
from torch.utils.data import Dataset

def generate_bounding_box():
    x1 = torch.rand(1)      # Random x1 coordinate
    y1 = torch.rand(1)      # Random y1 coordinate
    x2 = x1 + torch.rand(1) # Random x2 coordinate, x1 < x2
    y2 = y1 + torch.rand(1) # Random y2 coordinate, y1 < y2
    bbox = torch.cat((x1, y1, x2, y2), dim=0)
    return bbox


def generate_positive_negative(anchor):
    positive = generate_bounding_box()
    negative = generate_bounding_box()

    # Calculate IoU between anchor and positive/negative
    iou_positive = calculate_iou(anchor, positive)
    iou_negative = calculate_iou(anchor, negative)

    # Determine positive and negative based on IoU
    if iou_positive > iou_negative:
        return positive, negative
    else:
        return negative, positive


def calculate_iou(bbox1, bbox2):
    # Calculate the intersection area
    x1 = torch.max(bbox1[0], bbox2[0])
    y1 = torch.max(bbox1[1], bbox2[1])
    x2 = torch.min(bbox1[2], bbox2[2])
    y2 = torch.min(bbox1[3], bbox2[3])
    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)

    # Calculate the union area
    area_bbox1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area_bbox2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area_bbox1 + area_bbox2 - intersection

    # Calculate IoU
    iou = intersection / union
    return iou

class BoundingBoxDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples
        self.data = []
        for _ in range(self.num_samples):
            x1 = torch.rand(1)
            x2 = torch.rand(1)
            y1 = torch.rand(1)
            y2 = torch.rand(1)

            # Ensure x1 < x2 and y1 < y2
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            self.data.append([x1, y1, x2, y2])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        x1, y1, x2, y2 = self.data[index]
        anchor = torch.tensor([x1, y1, x2, y2])
        positive, negative = generate_positive_negative(anchor)
        # Calculate IoU between anchor and positive/negative
        iou_positive = calculate_iou(anchor, positive)
        iou_negative = calculate_iou(anchor, negative)
        return anchor, positive, negative, iou_positive, iou_negative


# Example usage
if __name__ == "__main__":
    dataset = BoundingBoxDataset(num_samples=1000)
    data = dataset[0]
    print(data)
