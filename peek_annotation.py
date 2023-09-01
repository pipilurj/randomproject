import json
with open("/home/pirenjie/data/coco/annotations/instances_val2017.json", "r") as f:
    anno = json.load(f)
# masks = [a["segmentation"] for a in anno["annotations"]]
# all_masks = []
# for mask in masks:
#     all_masks.extend(mask)
# maxlength = max([len(x)//2 for x in all_masks])
# pass
print("heyy")
# import pickle
#
# with open(f"train_mask.jsonl", 'rb') as f:
#     data = pickle.load(f)
#
# pass