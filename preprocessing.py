import pickle

import torch
import numpy as np

def load_data(file_path):
    with open(file_path, "rb") as file:
        data = pickle.load(file)
    return data


def prepare_data_for_torchmetrics(data_dict):
    """Convert your data format to the format expected by TorchMetrics"""

    preds = []
    targets = []

    for key in data_dict:
        # Prepare prediction data
        det_boxes = data_dict[key]["detic_data"]["boxes"]
        det_scores = data_dict[key]["detic_data"]["scores"]
        det_classes = data_dict[key]["detic_data"]["classes"]

        # Convert numpy arrays to torch tensors
        pred_dict = {
            "boxes": torch.tensor(det_boxes, dtype=torch.float32),
            "scores": torch.tensor(np.array(det_scores), dtype=torch.float32),
            "labels": torch.tensor(np.array(det_classes), dtype=torch.int64),
        }
        preds.append(pred_dict)

        # Prepare ground truth data
        gt_boxes = data_dict[key]["ground_truth"]["boxes"]
        gt_classes = data_dict[key]["ground_truth"]["classes"]

        target_dict = {
            "boxes": torch.tensor(gt_boxes, dtype=torch.float32),
            "labels": torch.tensor(np.array(gt_classes), dtype=torch.int64),
        }
        targets.append(target_dict)

    return preds, targets
